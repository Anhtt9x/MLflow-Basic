with open("artifacts/text.txt", "r") as file:
    s = file.read()
    print(s)


with open("artifacts/output.txt", "w") as file:
    file.write("We are learning DVC")
    print("Output create success")