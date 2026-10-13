import ast
import html
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp
from supabase import create_client

st.set_page_config(
    page_title="Khảo sát hàm số — Toán 12",
    page_icon="📈",
    layout="wide",
)

x = sp.Symbol("x", real=True)


# ---------- Đọc biểu thức toán an toàn ----------

def parse_math(text):
    tree = ast.parse(text.strip().replace("^", "**"), mode="eval")

    def read(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Chỉ nhập biểu thức toán học.")
            return sp.Rational(str(node.value))

        if isinstance(node, ast.Name):
            names = {"x": x, "e": sp.E, "E": sp.E, "pi": sp.pi}
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
                return left**right
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


def real_float(value):
    try:
        number = complex(sp.N(value))
        if abs(number.imag) < 1e-8 and math.isfinite(number.real):
            return float(number.real)
    except Exception:
        pass
    return None


def text_value(value):
    return sp.sstr(value)


def intervals_in(domain):
    if isinstance(domain, sp.Interval):
        return [domain]
    if isinstance(domain, sp.Union):
        parts = []
        for item in domain.args:
            parts.extend(intervals_in(item))
        return sorted(
            parts,
            key=lambda part: (
                -math.inf if part.start is -sp.oo else (real_float(part.start) or 0)
            ),
        )
    return []


def solve_roots(expression, domain):
    try:
        result = sp.solveset(expression, x, domain=domain)
        if result is sp.EmptySet:
            return [], None
        if isinstance(result, sp.FiniteSet):
            roots = [
                value for value in result
                if value.is_real is True and real_float(value) is not None
            ]
            return sorted(roots, key=lambda value: real_float(value)), None
        return [], "SymPy không biểu diễn được toàn bộ nghiệm thành một danh sách hữu hạn."
    except Exception:
        return [], "Chưa tính được đầy đủ nghiệm chính xác."


def safe_limit(expression, point, direction=None):
    try:
        if direction:
            return sp.limit(expression, x, point, dir=direction)
        return sp.limit(expression, x, point)
    except Exception:
        return None


def endpoint_value(expr, domain, point, side):
    if point in (sp.oo, -sp.oo):
        return safe_limit(expr, point)
    try:
        if domain.contains(point) is sp.true:
            return sp.simplify(expr.subs(x, point))
    except Exception:
        pass
    return safe_limit(expr, point, side)


def sign_of(expression, sample):
    try:
        value = complex(sp.N(expression.subs(x, sample)))
        if abs(value.imag) > 1e-7:
            return None
        if value.real > 1e-9:
            return 1
        if value.real < -1e-9:
            return -1
        return 0
    except Exception:
        return None


def sample_between(a, b):
    af, bf = real_float(a), real_float(b)
    if af is None and bf is None:
        return 0.0
    if af is None:
        return bf - max(1.0, abs(bf))
    if bf is None:
        return af + max(1.0, abs(af))
    return (af + bf) / 2


def format_limit(value):
    if value is None:
        return "chưa tính được"
    return text_value(value)


# ---------- Bảng biến thiên kiểu SGK ----------

def make_variation_table(expr, derivative, domain, critical_roots):
    pieces = intervals_in(domain)
    if not pieces:
        return "<p>Chưa dựng được bảng biến thiên cho miền này.</p>"

    critical_keys = {sp.sstr(root) for root in critical_roots}
    rows = [[], [], []]

    def add_separator():
        for row in rows:
            row.append('<td class="separator">║</td>')

    for piece_index, part in enumerate(pieces):
        if piece_index:
            add_separator()

        inside = []
        for root in critical_roots:
            try:
                if part.contains(root) is sp.true:
                    inside.append(root)
            except Exception:
                pass

        points = [part.start] + sorted(inside, key=lambda value: real_float(value)) + [part.end]

        for index, point in enumerate(points):
            if point is -sp.oo:
                x_label = "−∞"
            elif point is sp.oo:
                x_label = "+∞"
            else:
                x_label = text_value(point)

            excluded = False
            if point not in (-sp.oo, sp.oo):
                try:
                    excluded = domain.contains(point) is not sp.true
                except Exception:
                    excluded = False

            rows[0].append(
                '<td class="point">'
                + html.escape(x_label)
                + ("<br><small>không xác định</small>" if excluded else "")
                + "</td>"
            )

            if sp.sstr(point) in critical_keys:
                rows[1].append('<td class="zero">0</td>')
            elif excluded:
                rows[1].append('<td class="zero">||</td>')
            else:
                rows[1].append('<td></td>')

            side = "+" if point == part.start else "-"
            y_value = endpoint_value(expr, domain, point, side)
            rows[2].append(
                '<td class="value">' + html.escape(format_limit(y_value)) + "</td>"
            )

            if index < len(points) - 1:
                next_point = points[index + 1]
                sample = sample_between(point, next_point)
                sign = sign_of(derivative, sample)
                sign_text = "+" if sign == 1 else "−" if sign == -1 else "?"
                arrow = "↗" if sign == 1 else "↘" if sign == -1 else "→"

                rows[0].append('<td class="interval"></td>')
                rows[1].append(
                    '<td class="interval"><strong>' + sign_text + "</strong></td>"
                )
                rows[2].append(
                    '<td class="interval arrow">' + arrow + "</td>"
                )

    return """
    <div class="bbt-wrap">
      <table class="bbt">
        <tbody>
          <tr><th>x</th>""" + "".join(rows[0]) + """</tr>
          <tr><th>y′</th>""" + "".join(rows[1]) + """</tr>
          <tr><th>y</th>""" + "".join(rows[2]) + """</tr>
        </tbody>
      </table>
    </div>
    """


# ---------- Khảo sát ----------

def analyze(function_text, use_interval, left, right):
    expr = parse_math(function_text)
    derivative = sp.factor(sp.diff(expr, x))

    try:
        domain = sp.calculus.util.continuous_domain(expr, x, sp.S.Reals)
    except Exception:
        domain = sp.S.Reals

    if use_interval:
        domain = sp.Intersection(domain, sp.Interval(left, right))

    if domain is sp.EmptySet:
        raise ValueError("Khoảng khảo sát không giao với tập xác định của hàm.")

    critical_roots, critical_note = solve_roots(derivative, domain)
    x_intercepts, x_intercept_note = solve_roots(expr, domain)

    pieces = intervals_in(domain)
    monotonicity = []
    extrema = []

    for part in pieces:
        inside = []
        for root in critical_roots:
            try:
                if part.contains(root) is sp.true:
                    inside.append(root)
            except Exception:
                pass

        boundaries = [part.start] + sorted(
            inside, key=lambda value: real_float(value)
        ) + [part.end]

        signs = []
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            sign = sign_of(derivative, sample_between(a, b))
            signs.append(sign)
            if sign == 1:
                monotonicity.append(f"Đồng biến trên ({text_value(a)}; {text_value(b)}).")
            elif sign == -1:
                monotonicity.append(f"Nghịch biến trên ({text_value(a)}; {text_value(b)}).")

        for i, root in enumerate(inside):
            left_sign = signs[i] if i < len(signs) else None
            right_sign = signs[i + 1] if i + 1 < len(signs) else None
            y_value = sp.simplify(expr.subs(x, root))
            if left_sign == 1 and right_sign == -1:
                extrema.append(f"Cực đại tại x = {text_value(root)}, y = {text_value(y_value)}.")
            elif left_sign == -1 and right_sign == 1:
                extrema.append(f"Cực tiểu tại x = {text_value(root)}, y = {text_value(y_value)}.")

    limit_lines = []
    vertical_asymptotes = []

    for part in pieces:
        for point, direction, label in (
            (part.start, "+", "bên phải"),
            (part.end, "-", "bên trái"),
        ):
            if point in (-sp.oo, sp.oo):
                continue
            value = safe_limit(expr, point, direction)
            limit_lines.append(
                f"x → {text_value(point)} ({label}): f(x) → {format_limit(value)}"
            )
            if value in (sp.oo, -sp.oo):
                vertical_asymptotes.append(point)

    infinity_limits = []
    for point, label in ((-sp.oo, "−∞"), (sp.oo, "+∞")):
        value = safe_limit(expr, point)
        if value is not None:
            infinity_limits.append(f"x → {label}: f(x) → {text_value(value)}")

    asymptote_lines = []
    asymptote_specs = []
    for point in set(vertical_asymptotes):
        asymptote_lines.append(f"Tiệm cận đứng: x = {text_value(point)}.")
        pf = real_float(point)
        if pf is not None:
            asymptote_specs.append(("vertical", pf))

    for point, label in ((sp.oo, "+∞"), (-sp.oo, "−∞")):
        limit_value = safe_limit(expr, point)
        if limit_value is not None and limit_value.is_finite is True:
            asymptote_lines.append(
                f"Tiệm cận ngang khi x → {label}: y = {text_value(limit_value)}."
            )
            yf = real_float(limit_value)
            if yf is not None:
                asymptote_specs.append(("horizontal", yf))
            continue

        slope = safe_limit(expr / x, point)
        if slope is not None and slope.is_finite is True:
            intercept = safe_limit(expr - slope * x, point)
            if intercept is not None and intercept.is_finite is True and sp.simplify(slope) != 0:
                asymptote = sp.simplify(slope * x + intercept)
                line = f"Tiệm cận xiên khi x → {label}: y = {text_value(asymptote)}."
                if line not in asymptote_lines:
                    asymptote_lines.append(line)
                mf, bf = real_float(slope), real_float(intercept)
                if mf is not None and bf is not None:
                    asymptote_specs.append(("oblique", (mf, bf)))

    report = [
        f"Hàm số: y = {sp.sstr(expr)}",
        "",
        "1. TẬP XÁC ĐỊNH",
        f"D = {sp.sstr(domain)}",
        "",
        "2. SỰ BIẾN THIÊN",
        f"Đạo hàm: y′ = {sp.sstr(derivative)}",
    ]

    if critical_roots:
        report.append(
            "Nghiệm y′ = 0: " + ", ".join(text_value(r) for r in critical_roots) + "."
        )
    elif critical_note:
        report.append(critical_note)
    else:
        report.append("Không có nghiệm thực y′ = 0 trong miền khảo sát.")

    report.extend(monotonicity or ["Chưa xác định được các khoảng đơn điệu."])
    report.extend(extrema or ["Không có cực trị được xác định trong miền khảo sát."])

    if limit_lines or infinity_limits:
        report.append("Giới hạn:")
        report.extend(limit_lines)
        report.extend(infinity_limits)

    if asymptote_lines:
        report.append("Tiệm cận:")
        report.extend(asymptote_lines)
    else:
        report.append("Không tìm thấy tiệm cận.")

    report.extend(["", "3. GIAO ĐIỂM VỚI CÁC TRỤC"])
    if x_intercepts:
        report.append(
            "Giao với Ox: " + "; ".join(
                f"({text_value(root)}; 0)" for root in x_intercepts
            )
        )
    elif x_intercept_note:
        report.append(x_intercept_note)
    else:
        report.append("Không có giao điểm thực với Ox trong miền khảo sát.")

    try:
        if domain.contains(0) is sp.true:
            report.append(f"Giao với Oy: (0; {text_value(expr.subs(x, 0))})")
    except Exception:
        pass

    report.append("")
    report.append("4. BẢNG BIẾN THIÊN")
    bbt_html = make_variation_table(expr, derivative, domain, critical_roots)

    return (
        expr,
        derivative,
        domain,
        critical_roots,
        report,
        bbt_html,
        asymptote_specs,
    )


# ---------- Đồ thị ----------

def draw_plot(expr, domain, critical_roots, asymptote_specs, xmin, xmax):
    fig = go.Figure()
    function = sp.lambdify(x, expr, modules=["numpy"])

    for part in intervals_in(domain):
        start = -math.inf if part.start is -sp.oo else real_float(part.start)
        end = math.inf if part.end is sp.oo else real_float(part.end)

        if start is None or end is None:
            continue

        a = max(xmin, start)
        b = min(xmax, end)
        if a >= b:
            continue

        span = max(b - a, 1.0)
        if part.start not in (-sp.oo, sp.oo) and abs(a - start) < span * 1e-7:
            if domain.contains(part.start) is not sp.true:
                a += span * 1e-5
        if part.end not in (-sp.oo, sp.oo) and abs(b - end) < span * 1e-7:
            if domain.contains(part.end) is not sp.true:
                b -= span * 1e-5

        if a >= b:
            continue

        xs = np.linspace(a, b, 900)
        with np.errstate(all="ignore"):
            ys = np.asarray(function(xs), dtype=complex)

        if ys.ndim == 0:
            ys = np.full(xs.shape, ys, dtype=complex)

        valid = (
            np.isfinite(ys.real)
            & np.isfinite(ys.imag)
            & (np.abs(ys.imag) < 1e-7)
            & (np.abs(ys.real) < 1e8)
        )
        plotted_y = np.where(valid, ys.real, np.nan)

        # Ngắt đường tại bước nhảy lớn để tránh nối xuyên qua tiệm cận.
        jumps = np.where(np.abs(np.diff(plotted_y)) > max(20, span * 20))[0]
        for jump in jumps:
            plotted_y[jump + 1] = np.nan

        fig.add_trace(go.Scatter(
            x=xs,
            y=plotted_y,
            mode="lines",
            name="Đồ thị y = f(x)",
            line=dict(color="#3169e8", width=3),
            connectgaps=False,
            showlegend=False,
        ))

    point_x, point_y, point_labels = [], [], []
    for root in critical_roots:
        xv = real_float(root)
        yv = real_float(expr.subs(x, root))
        if xv is not None and yv is not None and xmin <= xv <= xmax:
            point_x.append(xv)
            point_y.append(yv)
            point_labels.append(f"({text_value(root)}; {text_value(sp.simplify(expr.subs(x, root)))})")

            fig.add_shape(
                type="line", x0=xv, x1=xv, y0=0, y1=yv,
                line=dict(color="#8993a4", width=1, dash="dot"),
            )
            fig.add_shape(
                type="line", x0=0, x1=xv, y0=yv, y1=yv,
                line=dict(color="#8993a4", width=1, dash="dot"),
            )

    if point_x:
        fig.add_trace(go.Scatter(
            x=point_x,
            y=point_y,
            text=point_labels,
            mode="markers+text",
            textposition="top right",
            name="Điểm tới hạn",
            marker=dict(color="#e34d59", size=10, line=dict(color="white", width=1)),
        ))

    for kind, value in asymptote_specs:
        if kind == "vertical":
            fig.add_vline(
                x=value, line_dash="dash", line_color="#8993a4",
                annotation_text=f"x = {value:g}",
            )
        elif kind == "horizontal":
            fig.add_hline(
                y=value, line_dash="dash", line_color="#8993a4",
                annotation_text=f"y = {value:g}",
            )
        elif kind == "oblique":
            slope, intercept = value
            fig.add_trace(go.Scatter(
                x=[xmin, xmax],
                y=[slope * xmin + intercept, slope * xmax + intercept],
                mode="lines",
                name="Tiệm cận xiên",
                line=dict(color="#8993a4", width=1.5, dash="dash"),
            ))

    fig.update_layout(
        title=dict(text=f"Đồ thị hàm số y = {sp.sstr(expr)}", x=0.02),
        template="plotly_white",
        height=520,
        margin=dict(l=35, r=25, t=65, b=40),
        hovermode="closest",
        xaxis=dict(
            title="x",
            range=[xmin, xmax],
            showgrid=True,
            gridcolor="#e8edf4",
            zeroline=True,
            zerolinecolor="#27364b",
            zerolinewidth=1.5,
            showline=True,
            linecolor="#667085",
        ),
        yaxis=dict(
            title="y",
            showgrid=True,
            gridcolor="#e8edf4",
            zeroline=True,
            zerolinecolor="#27364b",
            zerolinewidth=1.5,
            showline=True,
            linecolor="#667085",
        ),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "do-thi-khao-sat-ham-so",
                "scale": 2,
            },
        },
    )


# ---------- Kết nối Supabase và đăng nhập ----------

try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    if "supabase_client" not in st.session_state:
        st.session_state.supabase_client = create_client(supabase_url, supabase_key)
    supabase = st.session_state.supabase_client
except Exception:
    st.error("Chưa đọc được Secrets. Kiểm tra SUPABASE_URL và SUPABASE_KEY trong App settings → Secrets.")
    st.stop()

st.title("📈 Khảo sát hàm số — Toán 12")

if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "user_email" not in st.session_state:
    st.session_state.user_email = ""

with st.sidebar:
    st.subheader("Tài khoản")

    if st.session_state.user_id:
        st.write(f"Đang đăng nhập: {st.session_state.user_email}")
        if st.button("Đăng xuất"):
            try:
                supabase.auth.sign_out()
            except Exception:
                pass
            st.session_state.user_id = None
            st.session_state.user_email = ""
            st.rerun()
    else:
        mode = st.radio("Chọn thao tác", ["Đăng nhập", "Tạo tài khoản"])
        with st.form("auth_form"):
            email = st.text_input("Email")
            password = st.text_input("Mật khẩu", type="password")
            auth_submit = st.form_submit_button(mode)

        if auth_submit:
            try:
                if mode == "Đăng nhập":
                    response = supabase.auth.sign_in_with_password({
                        "email": email,
                        "password": password,
                    })
                    user = response.user
                    if user:
                        st.session_state.user_id = user.id
                        st.session_state.user_email = user.email or email
                        st.rerun()
                    else:
                        st.error("Đăng nhập chưa thành công. Hãy kiểm tra email và mật khẩu.")
                else:
                    response = supabase.auth.sign_up({
                        "email": email,
                        "password": password,
                    })
                    if response.session and response.user:
                        st.session_state.user_id = response.user.id
                        st.session_state.user_email = response.user.email or email
                        st.rerun()
                    else:
                        st.success("Đã tạo tài khoản. Hãy xác nhận email nếu Supabase yêu cầu, rồi đăng nhập.")
            except Exception as exc:
                st.error(f"Không đăng nhập/tạo tài khoản được: {exc}")

if not st.session_state.user_id:
    st.info("Hãy đăng nhập hoặc tạo tài khoản ở thanh bên để khảo sát và lưu lời giải.")
    st.stop()


# ---------- Nhập hàm và khảo sát ----------

st.subheader("Nhập hàm số")
with st.form("analysis_form"):
    function_text = st.text_input("Nhập f(x)", value="x^3 - 3*x + 1")
    student_name = st.text_input("Họ và tên học sinh")
    class_name = st.text_input("Lớp")

    use_interval = st.checkbox("Chỉ khảo sát trên một khoảng cụ thể")
    col1, col2 = st.columns(2)
    left = col1.number_input("Đầu trái khoảng khảo sát", value=-5.0)
    right = col2.number_input("Đầu phải khoảng khảo sát", value=5.0)

    col3, col4 = st.columns(2)
    plot_min = col3.number_input("Đồ thị: x nhỏ nhất", value=-5.0)
    plot_max = col4.number_input("Đồ thị: x lớn nhất", value=5.0)

    submitted = st.form_submit_button("Khảo sát hàm số")

if submitted:
    if plot_min >= plot_max:
        st.error("Giá trị x nhỏ nhất phải nhỏ hơn giá trị x lớn nhất.")
    elif use_interval and left >= right:
        st.error("Đầu trái của khoảng khảo sát phải nhỏ hơn đầu phải.")
    else:
        st.session_state.analysis_request = {
            "function_text": function_text,
            "student_name": student_name.strip(),
            "class_name": class_name.strip(),
            "use_interval": use_interval,
            "left": left,
            "right": right,
            "plot_min": plot_min,
            "plot_max": plot_max,
        }

request = st.session_state.get("analysis_request")

if request:
    try:
        (
            expr,
            derivative,
            domain,
            critical_roots,
            report,
            bbt_html,
            asymptote_specs,
        ) = analyze(
            request["function_text"],
            request["use_interval"],
            request["left"],
            request["right"],
        )

        st.markdown("### 1. Tập xác định và sự biến thiên")
        st.code("\n".join(report[:2]) if len(report) > 1 else report[0], language="text")

        st.markdown("### 2. Các bước khảo sát")
        st.code("\n".join(report[2:]), language="text")

        st.markdown("### 3. Bảng biến thiên đầy đủ")
        st.markdown(
            """
            <style>
            .bbt-wrap { overflow-x: auto; margin: 12px 0 22px; }
            table.bbt {
              border-collapse: collapse; min-width: 620px; width: 100%;
              font-family: Georgia, serif; background: white;
              border: 1px solid #8490a0; border-radius: 8px;
            }
            .bbt th, .bbt td {
              border: 1px solid #c4cad3; text-align: center;
              padding: 12px 10px; min-width: 65px;
            }
            .bbt th { background: #f2f5fa; width: 58px; }
            .bbt .interval { min-width: 75px; color: #44516a; }
            .bbt .arrow { font-size: 25px; color: #315de6; }
            .bbt .point { font-weight: 600; }
            .bbt .zero { font-weight: 700; }
            .bbt .separator {
              border-left: 4px double #505a69;
              border-right: 4px double #505a69;
              min-width: 18px; padding: 4px;
            }
            .bbt .value { min-width: 75px; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(bbt_html, unsafe_allow_html=True)

        st.markdown("### 4. Đồ thị")
        draw_plot(
            expr,
            domain,
            critical_roots,
            asymptote_specs,
            request["plot_min"],
            request["plot_max"],
        )

        if st.button("Lưu lời giải vào tài khoản"):
            if not request["student_name"] or not request["class_name"]:
                st.warning("Hãy nhập họ tên học sinh và lớp ở phía trên rồi khảo sát lại.")
            else:
                supabase.table("saved_solutions").insert({
                    "user_id": st.session_state.user_id,
                    "function_text": request["function_text"],
                    "solution_text": "\n".join(report),
                    "student_name": request["student_name"],
                    "class_name": request["class_name"],
                }).execute()
                st.success("Đã lưu lời giải kèm họ tên và lớp.")

    except Exception as exc:
        st.error(f"Không khảo sát được hàm số: {exc}")


# ---------- Lịch sử riêng từng học sinh; tài khoản quản trị xem tất cả ----------

st.divider()
st.subheader("Lời giải đã lưu")

try:
    admin_user_id = st.secrets.get("ADMIN_USER_ID", "")
    query = supabase.table("saved_solutions").select(
        "id, function_text, solution_text, created_at, student_name, class_name"
    )

    if st.session_state.user_id != admin_user_id:
        query = query.eq("user_id", st.session_state.user_id)

    response = query.order("created_at", desc=True).limit(100).execute()
    saved_rows = response.data or []

    if not saved_rows:
        st.caption("Bạn chưa lưu lời giải nào.")
    else:
        for row in saved_rows:
            title = (
                f"{row.get('student_name') or 'Chưa có tên'} — "
                f"{row.get('class_name') or 'Chưa có lớp'} — "
                f"{row.get('function_text') or 'Lời giải'}"
            )
            with st.expander(title):
                st.caption(f"Thời điểm lưu: {row.get('created_at', '')}")
                st.code(row.get("solution_text", ""), language="text")

except Exception as exc:
    st.error(f"Không tải được lịch sử: {exc}")
