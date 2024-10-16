from docxtpl import DocxTemplate

from lab3 import *
import numpy as np


def xlnx(x):
    return x * np.log(x)


payload = [xlnx, 2, 6]
methods = [rectangular_left, rectangular_right, rectangular_middle, trapezoid, simpson_integrator]

records = {}
for method in methods:
    result, h, steps = method(*payload)
    err = abs((22.8653 - result) / 22.8653)
    records[method.meta['name']] = {
        'result': f'{result:.6}',
        'h': f'{h:.2E}',
        'steps': steps,
        'err': f'{err:.2E}'
    }
    print(method.meta['name'])
    print(f"Результат - {result:.6}")
    print(f"Ширина шага - {h} | {steps} шагов")
    print(f"Погрешность - {err:.2E}", end='\n\n')

context = {
    "records": records
}

template = DocxTemplate("templates/ЛР_3_template.docx")
template.render(context)
template.save("out/ЛР_3.docx")
