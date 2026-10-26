import json


with open("students.json", "r") as file:
    students = json.load(file)


new_student = {
    "RollNo": 104,
    "Name": "David",
    "Marks": 78
}

students.append(new_student)

# Save the updated data into a new JSON file
with open("updated_student.json", "w") as file:
    json.dump(students, file, indent=4)

print("New record added successfully.")
print("Updated data saved in updated_students.json")
