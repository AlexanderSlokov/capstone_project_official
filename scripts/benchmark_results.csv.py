from fuzzywuzzy import fuzz
import pandas as pd

# Câu trả lời mẫu của bạn
reference_answer = """According to Article 123 of the Vietnamese Penal Code, the crime defined is murder. The maximum
penalty for this offense can be capital punishment (death penalty), depending on the severity of the act, aggravating
circumstances, and other factors related to the case."""

# Các câu trả lời thu thập được
answers = [
    """Based on the provided context, Article 123 of the Vietnamese Penal Code defines the crime of "Infringement
upon National Security." According to this article, any person who infringes upon Vietnam's national security
, including distorting the national border, or committing any other act for the purpose of infringing the national security of Socialist Republic of Vietnam, shall face the following penalties:
* The organizer or person whose activities cause serious consequences shall face a penalty of 12 - 20 years' imprisonment or life imprisonment.
* Any accomplice shall face a penalty of 05 - 15 years' imprisonment.
* Any person who makes preparation for the commitment of this criminal offence shall face a penalty of 01 - 05 years' imprisonment.
The maximum penalty for this crime is life imprisonment.""",
    """Based on the provided context, Article 123 of the Vietnamese Penal Code defines the crime of "Infringement
upon National Security." According to this article, any person who infringes upon Vietnam's national security
, including distorting the national border, or committing any other act for the purpose of infringing upon the national security of Socialist Republic of Vietnam, shall face the following penalties:

1. The organizer or person whose activities cause serious consequences shall face a penalty of 12 - 20 years' imprisonment or life imprisonment;

2. Any accomplice shall face a penalty of 05 - 15 years' imprisonment;

3. Any person who makes preparation for the commitment of this criminal offence shall face a penalty of 01 - 05 years' imprisonment.

The maximum penalty for this crime is 20 years' imprisonment or life imprisonment.""",
    """Based on the provided context, Article 123 of the Vietnamese Penal Code defines the crime of murder,
    which is punishable under the code. According to the article, "Whoever kills another person intentionally and
    without grounds shall face a penalty of 15 years' imprisonment to life imprisonment." Therefore, the maximum
    penalty for committing murder according to Article 123 of the Vietnamese Penal Code is life imprisonment.""",
    """Based on the provided context, Article 123 of the Vietnamese Penal Code defines the crime of "deliberate
    infliction of bodily harm upon another person." Specifically, the article states that a person who commits the
    crime of murder, as described in Clause 1 of this Article, shall face a penalty of 10 - 20 years' imprisonment,
    life imprisonment, or death.""",
    """> Answer:
  Based on the provided context, Article 123 of the Vietnamese Penal Code defines the crime of "Infringement upon National Security." According to this article, any person who infringes upon Vietnam's national security, including distorting the national border, or committing any other act for the purpose of infringing the national security of Socialist Republic of Vietnam, shall face the following penalties:
* For serious consequences, the organizer or perpetrator shall face a penalty of 12-20 years' imprisonment or life imprisonment.
* For less severe consequences, any accomplice shall face a penalty of 05-15 years' imprisonment.
* Preparation for the commission of this crime is punishable by a penalty of 01-05 years' imprisonment.
The maximum penalty for this crime is life imprisonment.""",
    """Based on the provided context, Article 123 of the Vietnamese Penal Code defines the crime of murder, which is punishable with the following penalties:
* For murder committed without any aggravating circumstances, the maximum penalty is 15-20 years' imprisonment.
* For murder committed with aggravating circumstances, such as the use of weapons or the killing of multiple persons, the maximum penalty is 20-30 years' imprisonment.
* For murder committed in a particularly cruel or heinous manner, the maximum penalty is life imprisonment or death.
It is important to note that these penalties are subject to change and may vary depending on the specific circumstances of the case and the discretion of the court."""
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
