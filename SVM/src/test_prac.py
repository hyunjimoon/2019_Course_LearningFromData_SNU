class Student:
    def __init__(self, name, marks):

        self.name = name
        self.marks = marks
    @property
    def gotmarks(self):
        return self.name + ' obtained ' + self.marks

s = Student("angie", "100")

print(s.gotmarks)