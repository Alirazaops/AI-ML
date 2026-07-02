import matplotlib.pyplot as plt

#Data
subject = ["Python", "Jave", "SQl", "Power BI"]
students = [40, 30, 20, 10]

plt.pie(students, labels=subject)

plt.title("Student Entolled")   

plt.show()
