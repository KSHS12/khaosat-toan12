import ast
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


def parse_math(text):
    """Đọc biểu thức bằng AST; không thực thi mã Python người dùng nhập."""
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


def set_intervals(domain):
    if isinstance(domain, sp.Interval):
        return [domain]
    if isinstance(domain, sp.Union):
        result = []
        for part in domain.args:
            result.extend(set_intervals(part))
        return result
    return []


def readable(value):
    return sp.sstr(value)


def real_number(value):
    try:
        number = complex(sp.N(value))
        if abs(number.imag) < 1e-9 and np.isfinite(number.real):
            return float(number.real)
    except Exception:
        pass
    return None


def solve_real_roots(expression, domain):
    try:
        result = sp.solveset(expression, x, domain=domain)
        if isinstance(result, sp.FiniteSet):
            roots = []
            for root in result:
                if root.is_real is True and real_number(root) is not None:
                    roots.append(root)
            return sorted(roots, key=lambda root: real_number(root)), None
        if result is sp.EmptySet:
            return [], None
        return [], "SymPy không biểu diễn được toàn bộ nghiệm thành một danh sách hữu hạn."
    except Exception:
        return [], "Không tính được đầy đủ nghiệm chính xác."


def analyze(text, use_interval, left, right):
    expr = parse_math(text)
    derivative = sp.diff(expr, x)

    try:
        domain = sp.calculus.util.continuous_domain(expr, x, sp.S.Reals)
    except Exception:
        domain = sp.S.Reals

    if use_interval:
        domain = sp.Intersection(domain, sp.Interval(left, right))

    if domain is sp.EmptySet:
        raise ValueError("Khoảng khảo sát không giao với tập xác định của hàm.")

    critical, critical_note = solve_real_roots(derivative, domain)
    zeros, zeros_note = solve_real_roots(expr, domain)

    lines = [
        f"Hàm số: y = {sp.sstr(expr)}",
        f"Tập xác định khi khảo sát: {sp.sstr(domain)}",
        f"Đạo hàm: f'(x) = {sp.sstr(derivative)}",
    ]

    if critical:
        lines.append("Nghiệm f'(x) = 0: " + ", ".join(readable(r) for r in critical))
    elif critical_note:
        lines.append(critical_note)
    else:
        lines.append("Không có nghiệm thực f'(x) = 0 trong miền đang khảo sát.")

    if zeros:
        lines.append("Giao điểm với trục Ox: " + ", ".join(
            f"({readable(r)}, 0)" for r in zeros
        ))
    elif zeros_note:
        lines.append(zeros_note)
    else:
        lines.append("Không có giao điểm thực với trục Ox trong miền đang khảo sát.")

    try:
        if sp.S.Zero in domain:
            lines.append(f"Giao điểm với trục Oy: (0, {sp.sstr(expr.subs(x, 0))})")
    except Exception:
        pass

    lines.append("")
    lines.append("Chiều biến thiên:")

    for part in set_intervals(domain):
        boundaries = [part.start]
        roots_inside = []
        for root in critical:
            try:
                if part.contains(root) is sp.true:
                    roots_inside.append(root)
            except Exception:
                pass
        boundaries.extend(roots_inside)
        boundaries.append(part.end)

        if len(boundaries) == 2 and sp.simplify(derivative) == 0:
            lines.append(f"Trên {sp.sstr(part)}: hàm số không đổi.")
            continue

        for a, b in zip(boundaries[:-1], boundaries[1:]):
            av, bv = real_number(a), real_number(b)
            if av is None and bv is None:
                sample = 0
            elif av is None:
                sample = bv - max(1, abs(bv))
            elif bv is None:
                sample = av + max(1, abs(av))
            else:
                sample = (av + bv) / 2

            try:
                sign = float(sp.N(derivative.subs(x, sample)))
                direction = "đồng biến" if sign > 0 else "nghịch biến" if sign < 0 else "cần xét thêm"
            except Exception:
                direction = "chưa xác định được"

            lines.append(f"Trên ({sp.sstr(a)}, {sp.sstr(b)}): {direction}.")

    return expr, domain, critical, "\n".join(lines)


def draw_plot(expr, domain, xmin, xmax, critical):
    fig = go.Figure()
    function = sp.lambdify(x, expr, modules=["numpy"])

    for part in set_intervals(domain):
        try:
            a = max(xmin, float(part.start)) if part.start is not -sp.oo else xmin
            b = min(xmax, float(part.end)) if part.end is not sp.oo else xmax
        except Exception:
            a, b = xmin, xmax

        if a >= b:
            continue

        xs = np.linspace(a, b, 700)
        with np.errstate(all="ignore"):
            ys = np.asarray(function(xs), dtype=complex)

        if ys.ndim == 0:
            ys = np.full(xs.shape, ys, dtype=complex)

        valid = (
            np.isfinite(ys.real)
            & np.isfinite(ys.imag)
            & (np.abs(ys.imag) < 1e-7)
            & (np.abs(ys.real) < 1e7)
        )
        plotted = np.where(valid, ys.real, np.nan)
        fig.add_trace(go.Scatter(
            x=xs, y=plotted, mode="lines", name="f(x)",
            line=dict(color="#315de6", width=3),
        ))

    points_x, points_y = [], []
    for root in critical:
        value = real_number(root)
        if value is not None and xmin <= value <= xmax:
            try:
                yvalue = real_number(expr.subs(x, root))
                if yvalue is not None:
                    points_x.append(value)
                    points_y.append(yvalue)
            except Exception:
                pass

    if points_x:
        fig.add_trace(go.Scatter(
            x=points_x, y=points_y, mode="markers", name="Điểm tới hạn",
            marker=dict(color="#e34d59", size=10),
        ))

    fig.update_layout(
        template="plotly_white",
        height=480,
        xaxis_title="x",
        yaxis_title="y",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


# Kết nối Supabase bằng Secrets của Streamlit.
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

# Đăng nhập / đăng ký
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
                else:
                    response = supabase.auth.sign_up({
                        "email": email,
                        "password": password,
                    })

                user = response.user
                if user:
                    st.session_state.user_id = user.id
                    st.session_state.user_email = user.email or email
                    st.success("Đăng nhập thành công.")
                    st.rerun()
                else:
                    st.info("Đã gửi yêu cầu tạo tài khoản. Nếu Supabase yêu cầu xác nhận email, hãy mở email để xác nhận rồi đăng nhập.")
            except Exception as exc:
                st.error(f"Không đăng nhập/tạo tài khoản được: {exc}")

if not st.session_state.user_id:
    st.info("Hãy đăng nhập hoặc tạo tài khoản ở thanh bên để sử dụng và lưu lời giải.")
    st.stop()


st.subheader("Nhập hàm số")
with st.form("analysis_form"):
    function_text = st.text_input("Nhập f(x)", value="x^3 - 3*x + 1")
    use_interval = st.checkbox("Chỉ khảo sát trên một khoảng cụ thể")
    c1, c2 = st.columns(2)
    left = c1.number_input("Đầu trái", value=-5.0)
    right = c2.number_input("Đầu phải", value=5.0)
    c3, c4 = st.columns(2)
    plot_min = c3.number_input("Đồ thị: x nhỏ nhất", value=-5.0)
    plot_max = c4.number_input("Đồ thị: x lớn nhất", value=5.0)
    submitted = st.form_submit_button("Khảo sát hàm số")

if submitted:
    if plot_min >= plot_max:
        st.error("Giá trị x nhỏ nhất của đồ thị phải nhỏ hơn giá trị lớn nhất.")
    elif use_interval and left >= right:
        st.error("Đầu trái của khoảng khảo sát phải nhỏ hơn đầu phải.")
    else:
        st.session_state.analysis_request = {
            "function_text": function_text,
            "use_interval": use_interval,
            "left": left,
            "right": right,
            "plot_min": plot_min,
            "plot_max": plot_max,
        }

request = st.session_state.get("analysis_request")

if request:
    try:
        expr, domain, critical, report = analyze(
            request["function_text"],
            request["use_interval"],
            request["left"],
            request["right"],
        )

        st.subheader("Kết quả khảo sát")
        st.code(report, language="text")
        st.subheader("Đồ thị")
        draw_plot(expr, domain, request["plot_min"], request["plot_max"], critical)

        if st.button("Lưu lời giải vào tài khoản"):
            supabase.table("saved_solutions").insert({
                "user_id": st.session_state.user_id,
                "function_text": request["function_text"],
                "solution_text": report,
            }).execute()
            st.success("Đã lưu lời giải.")

    except Exception as exc:
        st.error(f"Không khảo sát được hàm số: {exc}")


st.divider()
st.subheader("Lời giải đã lưu")

try:
    response = (
        supabase.table("saved_solutions")
        .select("id, function_text, solution_text, created_at")
        .eq("user_id", st.session_state.user_id)
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    )

    saved_rows = response.data or []
    if not saved_rows:
        st.caption("Bạn chưa lưu lời giải nào.")
    else:
        for row in saved_rows:
            title = f"{row.get('function_text', 'Lời giải')} — {row.get('created_at', '')}"
            with st.expander(title):
                st.code(row.get("solution_text", ""), language="text")
except Exception as exc:
    st.error(f"Không tải được lịch sử. Hãy kiểm tra quyền RLS của bảng: {exc}")
