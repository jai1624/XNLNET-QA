from bert import QA
from pathlib import Path

model = QA("model-oldversion")

doc = "GENERAL ADMINISTRATIVE MATTERS sections includes Working hours which describes The work hours of the Employee shall be from 09:00 AM to 06:30 PM for General Shift.The COMPANY shall have the authority to change the time schedule and shall be intimated to the Employee in advance.Working hours shorter than 04:00 shall be considered as Half Day for the purpose of wages."
q = 'what are the work hours of employees?'

answer = model.predict(doc,q)

print(answer['answer'])
# 1975
print(answer.keys())