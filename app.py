import ast
import math
import warnings

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from sympy.calculus.util import continuous_domain
from sympy.sets.sets import Set


st.set_page_config(
    page_title="Khảo sát hàm số — Toán 12",
    page_icon="📈",
    layout="wide",
)

x = sp.Symbol("x", real=True)
E = sp.E


# Chỉ cho phép các phép toán và hàm toán học trong danh sách này.
# Không dùng eval hoặc sympify trực tiếp trên nội dung người dùng nhập.
def parse_math(text):
    text = text.strip().replace("^", "**")
    tree = ast.parse(text, mode="eval")

    def read(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Chỉ nhập biểu thức toán học.")
            return sp.Rational(str(node.value))

        if isinstance(node, ast.Name):
            names = {
                "x": x,
                "e": E,
                "E": E,
                "pi": sp.pi,
            }
            if node.id not in names:
                raise ValueError(f"Ký hiệu chưa hỗ trợ: {node.id}")
            return names[node.id]

        if isinstance(node, ast.UnaryOp):
            value = read(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return value
            raise ValueError("Phép toán một ngôi chưa được hỗ trợ.")

        if isinstance(node, ast.BinOp):
            left, right = read(node.left), read(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            raise ValueError("Chỉ hỗ trợ cộng, trừ, nhân, chia và lũy thừa.")

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [read(arg) for arg in node.args]
            if node.keywords:
                raise ValueError("Không hỗ trợ tham số dạng từ khóa.")
            if node.func.id == "exp" and len(args) == 1:
                return sp.exp(args[0])
            if node.func.id == "sqrt" and len(args) == 1:
                return sp.sqrt(args[0])
            if node.func.id in ("log", "ln") and len(args) == 1:
                return sp.log(args[0])
            if node.func.id == "log" and len(args) == 2:
                return sp.log(args[0], args[1])
            raise ValueError("Hàm hỗ trợ: exp(...), sqrt(...), ln(...) và log(...).")

        raise ValueError("Biểu thức có ký hiệu hoặc cấu trúc chưa được hỗ trợ.")

    return read(tree.body)


def classify(expr):
    if expr.has(sp.log):
        return "Hàm có lôgarit"

    if expr.has(sp.exp) or any(
        p.exp.has(x) for p in expr.atoms(sp.Pow)
    ):
        return "Hàm mũ"

    try:
        combined = sp.together(expr)
        numerator, denominator = sp.fraction(combined)
        num_poly = sp.Poly(numerator, x)
        den_poly = sp.Poly(denominator, x)

        if den_poly.degree() == 0:
            degree = num_poly.degree()
            if degree <= 4:
                return f"Hàm đa thức bậc {degree}"
            return "Đa thức bậc lớn hơn 4 — ngoài phạm vi bản này"

        if den_poly.degree() == 1 and num_poly.degree() in (1, 2):
            return f"Hàm phân thức bậc {num_poly.degree()}/bậc 1"

        return "Hàm phân thức khác — cần kiểm tra phạm vi hỗ trợ"
    except (sp.PolynomialError, TypeError):
        return "Biểu thức toán học"


def intervals_in(domain):
    """Lấy các khoảng thành phần của một tập SymPy."""
    if isinstance(domain, sp.Interval):
        return [domain]
    if isinstance(domain, sp.Union):
        pieces = []
        for item in domain.args:
            pieces.extend(intervals_in(item))
        return pieces
    return []


def real_float(value):
    try:
        result = float(sp.N(value))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def readable(value):
    return sp.sstr(sp.simplify(value))


def test_sign(derivative, point):
    try:
        value = complex(sp.N(derivative.subs(x, point)))
        if not math.isfinite(value.real) or abs(value.imag) > 1e-8:
            return "chưa xác định"
        if value.real > 1e-8:
            return "đồng biến"
        if value.real < -1e-8:
            return "nghịch biến"
        return "đạo hàm bằng 0 hoặc gần 0"
    except Exception:
        return "chưa xác định"


def interval_sample(left, right):
    left_f, right_f = real_float(left), real_float(right)
    if left_f is None and right_f is None:
        return 0
    if left_f is None:
        return right_f - max(1.0, abs(right_f) * 0.1)
    if right_f is None:
        return left_f + max(1.0, abs(left_f) * 0.1)
    return (left_f + right_f) / 2


def solve_critical_points(derivative, domain):
    try:
        solutions = sp.solveset(derivative, x, domain=domain)
    except Exception:
        return None, "Không tìm được nghiệm đạo hàm bằng bộ giải ký hiệu."

    if isinstance(solutions, sp.FiniteSet):
        return sorted(
            [r for r in solutions if r.is_real is not False],
            key=lambda r: real_float(r) if real_float(r) is not None else 0,
        ), None

    if solutions is sp.EmptySet:
        return [], None

    return None, (
        "Bộ giải không đưa được toàn bộ nghiệm đạo hàm về một tập nghiệm hữu hạn. "
        f"Kết quả ký hiệu: {readable(solutions)}"
    )


def make_plot(expr, domain, plot_min, plot_max, critical_points):
    fig = go.Figure()
    plot_window = sp.Interval(sp.Float(plot_min), sp.Float(plot_max))
    visible_domain = domain.intersect(plot_window)

    try:
        fn = sp.lambdify(x, expr, modules=["numpy"])
    except Exception:
        st.warning("Không thể tạo đồ thị cho biểu thức này.")
        return

    for part in intervals_in(visible_domain):
        left = real_float(part.start)
        right = real_float(part.end)
        if left is None or right is None or left >= right:
            continue

        xs = np.linspace(left, right, 900)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                ys = np.asarray(fn(xs), dtype=float)
                if ys.ndim == 0:
                    ys = np.full_like(xs, float(ys))
            except Exception:
                continue

        ys[~np.isfinite(ys)] = np.nan

        # Chia đường vẽ ở các bước nhảy lớn để hạn chế nối xuyên qua tiệm cận.
        finite = ys[np.isfinite(ys)]
        if finite.size:
            lo, hi = np.percentile(finite, [3, 97])
            spread = max(hi - lo, 1.0)
            jumps = np.abs(np.diff(ys))
            split = jumps > spread * 0.45
            ys[1:][split] = np.nan

        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name="y = f(x)",
            line=dict(color="#315de6", width=3),
            connectgaps=False,
        ))

    marker_x, marker_y = [], []
    for root in critical_points or []:
        rx = real_float(root)
        if rx is None or rx < plot_min or rx > plot_max:
            continue
        try:
            ry = real_float(expr.subs(x, root))
            if ry is not None:
                marker_x.append(rx)
                marker_y.append(ry)
        except Exception:
            pass

    if marker_x:
        fig.add_trace(go.Scatter(
            x=marker_x,
            y=marker_y,
            mode="markers",
            name="Điểm tới hạn",
            marker=dict(color="#e34d59", size=9),
        ))

    fig.update_layout(
        template="plotly_white",
        height=430,
        margin=dict(l=20, r=20, t=25, b=20),
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(orientation="h", y=1.08),
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True)


def show_limits(expr, pieces):
    st.markdown("#### Giới hạn ở biên miền xác định")
    shown = set()

    for part in pieces:
        for endpoint, direction in ((part.start, "-"), (part.end, "+")):
            if endpoint in (sp.oo, -sp.oo):
                continue

            point = real_float(endpoint)
            if point is None:
                continue

            key = (str(endpoint), direction)
            if key in shown:
                continue
            shown.add(key)

            try:
                value = sp.limit(expr, x, endpoint, dir=direction)
                st.write(
                    f"x → {readable(endpoint)} ({'bên trái' if direction == '-' else 'bên phải'}): "
                    f"f(x) → {readable(value)}"
                )
                if value in (sp.oo, -sp.oo):
                    st.write(f"→ Tiệm cận đứng: x = {readable(endpoint)}")
            except Exception:
                st.write(f"Không tính được giới hạn tại x = {readable(endpoint)}.")

    for direction, label in ((sp.oo, "+∞"), (-sp.oo, "−∞")):
        try:
            value = sp.limit(expr, x, direction)
            if value.is_finite:
                st.write(f"Khi x → {label}: f(x) → {readable(value)}")
                st.write(f"→ Tiệm cận ngang: y = {readable(value)}")
            elif value in (sp.oo, -sp.oo):
                st.write(f"Khi x → {label}: f(x) → {readable(value)}")
        except Exception:
            pass


st.title("📈 Khảo sát hàm số — Toán 12")
st.caption(
    "Nhập biểu thức bằng x. Dùng * để nhân, ^ hoặc ** để lũy thừa. "
    "Ví dụ: x^3 - 3*x, (x^2+1)/(x-2), 2^x, log(x, 2)."
)

with st.form("survey"):
    expression_text = st.text_input(
        "Hàm số y = f(x)",
        value="x^3 - 3*x",
        help="Hàm đa thức tối đa bậc 4; phân thức bậc 1/1 hoặc 2/1; hàm mũ và lôgarit.",
    )

    use_interval = st.checkbox("Chỉ khảo sát trên một khoảng cụ thể")
    col1, col2 = st.columns(2)
    with col1:
        interval_left = st.number_input("Đầu trái khoảng", value=-5.0)
    with col2:
        interval_right = st.number_input("Đầu phải khoảng", value=5.0)

    st.markdown("**Khoảng hiển thị đồ thị**")
    col3, col4 = st.columns(2)
    with col3:
        plot_min = st.number_input("x nhỏ nhất trên đồ thị", value=-10.0)
    with col4:
        plot_max = st.number_input("x lớn nhất trên đồ thị", value=10.0)

    submitted = st.form_submit_button("Khảo sát hàm số", type="primary")


if submitted:
    try:
        if not expression_text.strip():
            raise ValueError("Hãy nhập hàm số.")

        if plot_min >= plot_max:
            raise ValueError("Khoảng hiển thị đồ thị không hợp lệ.")

        expr = parse_math(expression_text)
        kind = classify(expr)

        if "ngoài phạm vi" in kind or "cần kiểm tra" in kind:
            st.error(
                f"{kind}. Bản này chỉ nhận đa thức bậc không quá 4, "
                "phân thức bậc 1/1 hoặc 2/1, hàm mũ và hàm lôgarit."
            )
            st.stop()

        try:
            domain = continuous_domain(expr, x, sp.S.Reals)
        except Exception as exc:
            st.error(f"Không xác định được tập xác định: {exc}")
            st.stop()

        if use_interval:
            if interval_left >= interval_right:
                st.error("Đầu trái khoảng phải nhỏ hơn đầu phải.")
                st.stop()
            survey_domain = domain.intersect(
                sp.Interval(sp.Float(interval_left), sp.Float(interval_right))
            )
        else:
            survey_domain = domain

        st.subheader("Kết quả")
        st.write(f"**Dạng hàm:** {kind}")
        st.latex(r"y = " + sp.latex(expr))
        st.write("**Tập xác định:**")
        st.latex(r"D = " + sp.latex(domain))

        if use_interval:
            st.write(
                f"**Phần tập xác định đang khảo sát:** "
                f"{readable(survey_domain)}"
            )

        derivative = sp.simplify(sp.diff(expr, x))
        second_derivative = sp.simplify(sp.diff(expr, x, 2))

        st.markdown("#### Đạo hàm")
        st.latex(r"y' = " + sp.latex(derivative))
        st.latex(r"y'' = " + sp.latex(second_derivative))

        roots, root_note = solve_critical_points(derivative, survey_domain)
        if root_note:
            st.warning(root_note)
            roots_for_plot = []
        else:
            roots_for_plot = roots
            st.markdown("#### Điểm tới hạn và cực trị")

            if not roots:
                st.write("Không có nghiệm y′ = 0 trong miền đang khảo sát.")
            else:
                for root in roots:
                    value = sp.simplify(expr.subs(x, root))
                    st.write(
                        f"x = **{readable(root)}**, "
                        f"f(x) = **{readable(value)}**"
                    )

        pieces = intervals_in(survey_domain)
        st.markdown("#### Chiều biến thiên")
        if not pieces:
            st.write("Không biểu diễn được miền khảo sát thành các khoảng.")
        else:
            for part in pieces:
                relevant_roots = []
                if not roots: st.write("Không có nghiệm y′ = 0 trong miền đang khảo sát.")
...                 boundaries = [part.start, *relevant_roots, part.end]
...                 st.write(f"**Trên khoảng {readable(part)}:**")
... 
...                 if len(boundaries) == 2 and derivative == 0:
...                     st.write("Hàm số không đổi trên khoảng này.")
...                     continue
... 
...                 for left, right in zip(boundaries[:-1], boundaries[1:]):
...                     sample = interval_sample(left, right)
...                     direction = test_sign(derivative, sample)
...                     st.write(
...                         f"• ({readable(left)}, {readable(right)}): {direction}"
...                     )
... 
...         show_limits(expr, pieces)
... 
...         st.markdown("#### Đồ thị")
...         make_plot(expr, survey_domain, plot_min, plot_max, roots_for_plot)
... 
...         st.info(
...             "Đạo hàm, giới hạn và nghiệm được tính bằng SymPy. "
...             "Đồ thị là hình vẽ số để minh họa; hãy đối chiếu bảng biến thiên "
...             "với bài giải khi dùng làm tài liệu giảng dạy."
...         )
... 
...     except (ValueError, SyntaxError, ZeroDivisionError) as exc:
...         st.error(f"Không xử lý được biểu thức: {exc}")
...     except Exception as exc:
...         st.error(f"Có lỗi khi khảo sát hàm: {exc}")
... 
... else:
...     st.info("Nhập hàm số, chọn khoảng nếu cần, rồi nhấn “Khảo sát hàm số”.")
