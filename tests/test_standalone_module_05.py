import sys
import time
import argparse
from pathlib import Path

# Hỗ trợ hiển thị tiếng Việt UTF-8 trên terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm thư mục backend vào sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Chỉ import DUY NHẤT Module 5 (IntentRouter) và IntentService
from app.services.intent_router import (
    intent_router,
    STRATEGY_PROCEDURAL_EXTRACTIVE,
    STRATEGY_TARGETED_QA
)
from app.services.intent_service import intent_service

__test__ = False


SAMPLE_ROUTER_CASES = [
    ("Hướng dẫn các bước đăng ký tài khoản doanh nghiệp mới", "Quy trình Đăng ký (Mode A)"),
    ("Làm thế nào để đổi mật khẩu tài khoản?", "Quy trình Đổi mật khẩu (Mode A)"),
    ("Các bước nộp báo cáo định kỳ tai nạn lao động", "Quy trình Báo cáo TNLĐ (Mode A)"),
    ("Hướng dẫn nộp báo cáo an toàn vệ sinh lao động", "Quy trình Báo cáo ATVSLĐ (Mode A)"),
    ("Làm sao để sửa thông tin doanh nghiệp?", "Quy trình Đổi TT Doanh nghiệp (Mode A)"),
    ("Tôi muốn xem số liệu thống kê các vụ tai nạn", "Quy trình Thống kê (Mode A)"),
    ("Sau khi đăng ký thì ai là người kích hoạt tài khoản?", "Factoid Đăng ký (Mode B - có trong HDSD)"),
    ("Trường Tổng quỹ lương khi báo cáo TNLĐ nhập đơn vị là gì?", "Factoid Báo cáo TNLĐ (Mode B - có trong HDSD)"),
    ("Hạn nộp báo cáo an toàn vệ sinh lao động là ngày mấy?", "Factoid Báo cáo ATVSLĐ (Mode B - có trong HDSD)"),
    ("Mức phạt vi phạm hành chính khi không nộp báo cáo là bao nhiêu tiền?", "Factoid Ngách ngoài HDSD (Mode B - Factoid_nanswer)"),
    ("Cho tôi số tổng đài hotline và zalo hỗ trợ", "Ngoài 7 phân hệ -> Target: None (Do Module 4 xử lý)"),
]


def test_module_05(query: str, verbose: bool = True):
    """Kiểm tra độc lập 100% logic của Module 5 (Strategy & Domain Router)."""
    start_time = time.perf_counter()
    
    # Phân loại chiến lược và phân hệ mục tiêu qua Single-Pass Router
    result = intent_service.classify_intent(query)
    strategy, target_module = result.strategy, result.target_module
    
    # Làm sạch từ đệm đàm thoại cho BM25
    cleaned_query = intent_router.strip_conversational_noise(query)
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    strat_label = (
        "🚀 MODE A: PROCEDURAL_EXTRACTIVE (Trích xuất nguyên văn các bước < 50ms)"
        if strategy == STRATEGY_PROCEDURAL_EXTRACTIVE
        else "🧠 MODE B: TARGETED_QA (Gửi Context tới Qwen LLM tổng hợp 1-3 câu hoặc thẻ [Factoid_nanswer])"
    )
    
    if verbose:
        print("\n" + "=" * 75)
        print(f"📥 INPUT QUERY:          \"{query}\"")
        print("-" * 75)
        print(f"🏷️ MACRO INTENT:         [{result.intent}]")
        print(f"🏢 TARGET MODULE:        {target_module if target_module else 'Không xác định (Toàn cục)'}")
        print(f"🔀 RETRIEVAL STRATEGY:   {strat_label}")
        print(f"💡 AI REASONING:         \"{result.matched_exemplar}\"")
        print(f"🧹 BM25 CLEANED QUERY:   \"{cleaned_query}\"")
        print(f"⏱️ EXECUTION TIME:       {elapsed_ms:.4f} ms")
        print("=" * 75)

    return strategy, target_module, elapsed_ms


def run_sample_tests():
    """Chạy bộ mẫu kiểm thử tiêu chuẩn của Module 5."""
    print("=" * 80)
    print("🧪 BẮT ĐẦU CHẠY BỘ KIỂM THỬ MẪU MODULE 5 (SINGLE-PASS STRATEGY & DOMAIN ROUTER)")
    print("=" * 80)
    
    latencies = []
    for idx, (query, desc) in enumerate(SAMPLE_ROUTER_CASES, 1):
        print(f"\n[{idx}/{len(SAMPLE_ROUTER_CASES)}] Scenario: {desc}")
        _, _, lat = test_module_05(query, verbose=True)
        latencies.append(lat)
        
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    print("\n" + "=" * 80)
    print(f"✅ HOÀN TẤT {len(SAMPLE_ROUTER_CASES)} CASES KIỂM THỬ MẪU MODULE 5")
    print(f"⚡ Độ trễ trung bình: {avg_lat:.2f} ms")
    print("=" * 80 + "\n")


def interactive_chat():
    """Chế độ nhập Terminal trực tiếp để test Module 5."""
    print("=" * 70)
    print("💬 CHẾ ĐỘ TEST ĐỘC LẬP MODULE 5: SINGLE-PASS STRATEGY & DOMAIN ROUTER")
    print("=" * 70)
    print("👉 Gõ 'sample' để chạy bộ 11 kiểm thử mẫu tiêu chuẩn.")
    print("👉 Gõ 'exit' hoặc 'q' để thoát.\n")

    while True:
        try:
            user_input = input("👉 Nhập Quest: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Đã thoát khỏi chương trình test Module 5.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "q", "/exit"]:
            print("👋 Đã thoát khỏi chương trình test Module 5.")
            break
            
        if user_input.lower() == "sample":
            run_sample_tests()
            continue

        test_module_05(user_input, verbose=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Độc lập Module 5 - Strategy & Domain Router")
    parser.add_argument("-q", "--query", type=str, help="Câu hỏi đơn lẻ cần kiểm tra")
    parser.add_argument("--sample", action="store_true", help="Chạy toàn bộ 11 ca kiểm thử mẫu tiêu chuẩn")
    args = parser.parse_args()

    if args.sample:
        run_sample_tests()
    elif args.query:
        test_module_05(args.query, verbose=True)
    else:
        interactive_chat()
