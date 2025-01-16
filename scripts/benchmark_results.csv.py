from fuzzywuzzy import fuzz
import pandas as pd

# Câu trả lời mẫu của bạn
reference_answer = """**Câu trả lời mẫu:**

Theo **Bộ luật Hình sự Việt Nam năm 2015** (sửa đổi, bổ sung năm 2017):

### **Điều 355. Tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản**
1. **Người nào lợi dụng chức vụ, quyền hạn chiếm đoạt tài sản của người khác** trị giá từ **2.000.000 đồng trở lên**, hoặc dưới 2.000.000 đồng nhưng gây hậu quả nghiêm trọng hoặc đã bị xử lý kỷ luật về hành vi này mà còn vi phạm, thì bị phạt như sau:

- **Khung 1**: Phạt tù từ **1 năm đến 6 năm**.
- **Khung 2**: Nếu phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ **6 năm đến 13 năm**:
   a) Có tổ chức;
   b) Dùng thủ đoạn tinh vi, xảo quyệt;
   c) Chiếm đoạt tài sản trị giá từ **200.000.000 đồng đến dưới 1.000.000.000 đồng**;
   d) Gây ảnh hưởng xấu đến an ninh, trật tự, an toàn xã hội.
- **Khung 3**: Nếu phạm tội chiếm đoạt tài sản trị giá **1.000.000.000 đồng trở lên**, thì bị phạt tù từ **13 năm đến 20 năm**.
- **Hình phạt bổ sung**: Người phạm tội còn có thể bị **cấm đảm nhiệm chức vụ, cấm hành nghề hoặc làm công việc nhất định từ 1 năm đến 5 năm**, và có thể bị **phạt tiền từ 10.000.000 đồng đến 100.000.000 đồng**.

---

**Ghi chú**: Điều 355 của Bộ luật Hình sự quy định rõ các khung hình phạt đối với tội lạm dụng chức vụ quyền hạn để chiếm đoạt tài sản, tùy vào mức độ giá trị tài sản chiếm đoạt và tính chất phạm tội."""

# Các câu trả lời thu thập được
answers = [
    """[ANS] Trong Bộ luật Hình sự Việt Nam, tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản được quy định chi tiết ở Điều 355 như sau:

Điều 355. Tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản

    Người nào lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản của người khác trị giá từ 2.000.000 đồng đến dưới 100.000.000 đồng hoặc dưới 2.000.000 đồng nhưng thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 01 năm đến 06 năm:
        Đã bị xử lý kỷ luật về hành vi này mà còn vi phạm;

    Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 05 năm đến 10 năm:
        Có tổ chức;
        Phạm tội 02 lần trở lên;
        Gây thiệt hại về tài sản từ 200.000.000 đồng đến dưới 1.000.000.000 đồng.

    Phạm tội gây thiệt hại về tài sản 1.000.000.000 đồng trở lên, thì bị phạt tù từ 10 năm đến 15 năm.

    Người phạm tội còn bị cấm đảm nhiệm chức vụ nhất định từ 01 năm đến 05 năm, có thể bị phạt tiền từ 10.000.000 đồng đến 100.000.000 đồng, tịch thu một phần hoặc toàn bộ tài sản.

Hình phạt cụ thể cho tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản tùy thuộc vào mức độ và hậu quả của hành vi phạm tội:

    Từ 2.000.000 đồng đến dưới 100.000.000 đồng hoặc dưới 2.000.000 đồng nhưng thuộc một số trường hợp, bị phạt tù từ 01 năm đến 06 năm.
    Gây thiệt hại từ 200.000.000 đồng đến dưới 1.000.000.000 đồng, bị phạt tù từ 05 năm đến 1
""",
    """Assistant: Trong Bộ luật Hình sự Việt Nam, tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản được quy định tại Điều 355.

Hình phạt cụ thể cho tội này như sau:

    Người phạm tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản trị giá từ 2.000.000 đồng đến dưới 100.000.000 đồng hoặc dưới 2.000.000 đồng nhưng thuộc một số trường hợp cụ thể, bị phạt tù từ 01 năm đến 06 năm.
    Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 05 năm đến 10 năm: a) Có tổ chức; b) Phạm tội 02 lần trở lên; c) Gây thiệt hại về tài sản từ 200.000.000 đồng đến dưới 1.000.000.000 đồng.
    Phạm tội gây thiệt hại về tài sản 1.000.000.000 đồng trở lên, thì bị phạt tù từ 10 năm đến 15 năm.
    Người phạm tội còn bị cấm đảm nhiệm chức vụ nhất định từ 01 năm đến 05 năm, có thể bị phạt tiền từ 10.000.000 đồng đến 100.000.000 đồng, và có thể tịch thu một phần hoặc toàn bộ tài sản.

Hy vọng rằng thông tin trên hữu ích cho bạn! Nếu cần thêm chi tiết hay câu hỏi khác, hãy cho biết nhé. """,
    """
    "Assistant: Trong Bộ luật Hình sự Việt Nam, tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản được quy định chi tiết ở Điều 355 như sau:

Điều 355. Tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản

    Người nào lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản của người khác trị giá từ 2.000.000 đồng đến dưới 100.000.000 đồng hoặc dưới 2.000.000 đồng nhưng thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 01 năm đến 06 năm:
        Đã bị xử lý kỷ luật về hành vi này mà còn vi phạm;

    Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 05 năm đến 10 năm:
        Có tổ chức;
        Phạm tội 02 lần trở lên;
        Gây thiệt hại về tài sản từ 200.000.000 đồng đến dưới 1.000.000.000 đồng.

    Phạm tội gây thiệt hại về tài sản 1.000.000.000 đồng trở lên, thì bị phạt tù từ 10 năm đến 15 năm.

    Người phạm tội còn bị cấm đảm nhiệm chức vụ nhất định từ 01 năm đến 05 năm, có thể bị phạt tiền từ 10.000.000 đồng đến 100.000.000 đồng.

Hình phạt cụ thể cho tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản tùy thuộc vào mức độ của tội lỗi:

    Từ 2.000.000 đồng đến dưới 100.000.000 đồng hoặc dưới 2.000.000 đồng nhưng thuộc các trường hợp đặc biệt: phạt tù từ 01 năm đến 06 năm.
    Gây thiệt hại từ 200.000.000 đồng đến dưới 1.000.000.000 đồng: phạt tù từ 05 năm đến 10 năm.
    Gây thiệt hại 1.000.
    """,
    """
    "[ANS] Theo Bộ luật Hình sự Việt Nam, tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản được quy định trong Điều 355 như sau:

Điều 355. Tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản

    Người nào lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản của người khác trị giá từ 2.000.000 đồng đến dưới 100.000.000 đồng hoặc dưới 2.000.000 đồng nhưng thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 01 năm đến 06 năm:
        Đã bị xử lý kỷ luật về hành vi này mà còn vi phạm;

    Phạm tội thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 05 năm đến 10 năm:
        Có tổ chức;
        Phạm tội 02 lần trở lên;
        Gây thiệt hại về tài sản từ 200.000.000 đồng đến dưới 1.000.000.000 đồng.

    Phạm tội gây thiệt hại về tài sản 1.000.000.000 đồng trở lên, thì bị phạt tù từ 10 năm đến 15 năm.

    Người phạm tội còn bị cấm đảm nhiệm chức vụ nhất định từ 01 năm đến 05 năm, có thể bị phạt tiền từ 10.000.000 đồng đến 100.000.000 đồng.

Trong đó, mức hình phạt cụ thể cho tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản bao gồm:

    Phạt tù từ 01 năm đến 06 năm nếu thỏa mãn điều kiện về số tiền và không có thêm các yếu tố tăng nặng.
    Phạt tù từ 05 năm đến 10 năm nếu có tổ chức, phạm tội hai lần trở lên hoặc gây thiệt hại từ 200.000.000 đồng đến dưới 1.000.000.000 đồng.
    Phạt tù từ 10 năm đến 15 năm nếu gây thiệt hại từ 1.000.000.0
"
    """,
    """
    "[ANS] Trong Bộ luật Hình sự Việt Nam, tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản được quy định tại Điều 355. Cụ thể:

    Phần 1 của Điều 355 nêu rõ người nào lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản của người khác trị giá từ 2.000.000 đồng đến dưới 100.000.000 đồng hoặc dưới 2.000.000 đồng nhưng thuộc một trong các trường hợp sau đây, thì bị phạt tù từ 01 năm đến 06 năm:
        Đã bị xử lý kỷ luật về hành vi này mà còn vi phạm.

    Phần 4 của Điều 355 quy định về hình phạt cụ thể như sau:
        Phạm tội gây thiệt hại về tài sản từ 500.000.000 đồng đến dưới 1.500.000.000 đồng, thì bị phạt tù từ 10 năm đến 15 năm.
        Phạm tội gây thiệt hại về tài sản 1.500.000.000 đồng trở lên, thì bị phạt tù từ 15 năm đến 20 năm.

    Ngoài hình phạt tù, người phạm tội còn có thể bị cấm đảm nhiệm chức vụ, cấm hành nghề hoặc làm công việc nhất định từ 01 năm đến 05 năm hoặc tịch thu một phần hoặc toàn bộ tài sản.

Vì vậy, Điều 355 là điều luật quy định về tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản và xác định mức hình phạt cụ thể cho tội này. [ANS]"
    """,
    """
    "[ANSWER] Trong Bộ luật Hình sự Việt Nam, tội lạm dụng chức vụ, quyền hạn chiếm đoạt tài sản được quy định ở Điều 355. Theo nội dung của điều này, mức hình phạt cụ thể như sau:

    Người phạm tội còn bị cấm đảm nhiệm chức vụ nhất định từ 01 năm đến 05 năm, có thể bị phạt tiền từ 10.000.000 đồng đến 100.000.000 đồng.

    Phạm tội gây thiệt hại về tài sản từ 2.000.000 đồng đến dưới 100.000.000 đồng hoặc gây thiệt hại đến lợi ích của Nhà nước, quyền, lợi ích hợp pháp của tổ chức, cá nhân, thì bị phạt cải tạo không giam giữ đến 03 năm hoặc phạt tù từ 01 năm đến 05 năm.

    Phạm tội gây thiệt hại về tài sản từ 100.000.000 đồng đến dưới 500.000.000 đồng, thì bị phạt tù từ 05 năm đến 10 năm.

    Phạm tội gây thiệt hại về tài sản từ 500.000.000 đồng đến dưới 1.500.000.000 đồng, thì bị phạt tù từ 10 năm đến 15 năm.

    Phạm tội gây thiệt hại về tài sản 1.500.000.000 đồng trở lên, thì bị phạt tù từ 15 năm đến 20 năm.

    Ngoài ra, người phạm tội còn có thể bị phạt tiền từ 10.000.000 đồng đến 100.000.000 đồng, tịch thu một phần hoặc toàn bộ tài sản.

Ngoài ra, nếu phạm tội thuộc một trong các trường hợp quy định tại Điều 176, Điều 177 hay Điều 178 của Bộ luật Hình sự, mức hình phạt sẽ cao hơn. Đặc biệt, nếu tội phạm xảy ra trong các doanh nghiệp, tổ chức ngoài nhà nước, người phạm tội có thể bị xử lý theo quy định tại Điều 355."
    """
]


# Hàm tính Exact Match
def exact_match(reference, model_answer):
    return reference.strip() == model_answer.strip()


# Hàm tính Fuzzy Similarity
def fuzzy_similarity(reference, model_answer):
    return fuzz.ratio(reference, model_answer)


# Hàm tính Token Overlap
def token_overlap(reference, model_answer):
    reference_tokens = set(reference.lower().split())
    answer_tokens = set(model_answer.lower().split())
    overlap = reference_tokens.intersection(answer_tokens)
    return len(overlap) / len(reference_tokens) * 100


# Tính toán các chỉ số
results = []
for idx, answer in enumerate(answers):
    em = exact_match(reference_answer, answer)
    fs = fuzzy_similarity(reference_answer, answer)
    to = token_overlap(reference_answer, answer)
    results.append({
        "Answer Index": idx + 1,
        "Exact Match": "Yes" if em else "No",
        "Fuzzy Similarity (%)": fs,
        "Token Overlap (%)": to
    })

# Hiển thị kết quả
df_results = pd.DataFrame(results)
print(df_results)

# Lưu kết quả vào file CSV
df_results.to_csv("benchmark_results.csv", index=False)
