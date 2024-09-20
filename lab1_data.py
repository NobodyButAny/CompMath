from lab1 import *
from docxtpl import DocxTemplate

function = DiffFunction(
    lambda x: 2 * math.log(x) - 1 / x,
    lambda x: 2 / x + 1 / (x ** 2),
    lambda x: -2 / (x ** 2) - 2 / (x ** 3)
)


def format_float(num: float):
    return '{:.9f}'.format(num)


methods = [newton, chords, secants, finite_sum_newton, stephensen, simple_iter]

setup = {
    "f": function,
    "a": 0.1,
    "b": 2,
    "epsilon": 10 ** (-7)
}

records = {}
for method in methods:
    solution = method(**setup)
    record = {
        "interval": [0.1, 2],
        "solution": format_float(solution[0]),
        "iterations": list(map(format_float, solution[1])),
        "iterations_n": len(solution[1]),
        "error": abs(solution[1][-1] - solution[1][-2])
    }
    records[method.meta['name']] = record

print(records)

context = {
    "records": records
}

template = DocxTemplate("templates/ЛР_1_template.docx")
template.render(context)
template.save("out/ЛР_1.docx")
