import torch


def generate_ans(id_list, ans_list):
    result = []
    for a in range(0, len(id_list)):
        idx = id_list[a]
        ans = ans_list[a]
        if len(ans) == 4:
            ans = [["A", "B", "C", "D"][int(torch.max(ans, dim=0)[1])]]
        else:
            ans_ = int(torch.max(ans, dim=0)[1])
            ans = []
            for x, y in [(8, "D"), (4, "C"), (2, "B"), (1, "A")]:
                if ans_ >= x:
                    ans.append(y)
                    ans_ -= x
        result.append({"id": idx, "answer": ans})

    return result
