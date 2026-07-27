# Library Management System

library = {}

while True:
    print("\n===== LIBRARY MANAGEMENT SYSTEM =====")
    print("1. Add Book")
    print("2. Display Books")
    print("3. Issue Book")
    print("4. Return Book")
    print("5. Search Book")
    print("6. Exit")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        book = input("Enter book name: ")

        if book in library:
            print("Book already exists!")
        else:
            library[book] = "Available"
            print("Book added successfully!")

    elif choice == 2:
        if len(library) == 0:
            print("Library is empty.")
        else:
            print("\nBooks in Library:")
            for book, status in library.items():
                print(book, "-", status)

    elif choice == 3:
        book = input("Enter book name to issue: ")

        if book in library:
            if library[book] == "Available":
                library[book] = "Issued"
                print("Book issued successfully!")
            else:
                print("Book is already issued.")
        else:
            print("Book not found.")

    elif choice == 4:
        book = input("Enter book name to return: ")

        if book in library:
            if library[book] == "Issued":
                library[book] = "Available"
                print("Book returned successfully!")
            else:
                print("Book was not issued.")
        else:
            print("Book not found.")

    elif choice == 5:
        book = input("Enter book name to search: ")

        if book in library:
            print(book, "is", library[book])
        else:
            print("Book not found.")

    elif choice == 6:
        print("Thank you for using the Library Management System!")
        break

    else:
        print("Invalid choice! Please try again.")