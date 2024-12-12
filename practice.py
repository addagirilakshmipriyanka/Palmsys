def stu_attendance():
    a_date = datetime.today().strftime("%d-%m-%Y")
    a_time = datetime.now().strftime("%H:%M:%S")
    current_time = datetime.strptime(a_time, "%H:%M:%S").time()

    # Retrieve classes for the current date
    classes_today = list(mongo.db.Class.find({'date': a_date}))

    if not classes_today:
        return jsonify({"error": "No classes found for today."})

    class_scheduled = False

    for class_info in classes_today:
        try:
            # Parse start and end times
            start_time = parse_time(class_info['start_time'])
            end_time = parse_time(class_info['end_time'])

            # Check if current time is within the class schedule
            if start_time <= current_time <= end_time:
                class_scheduled = True
                class_id = class_info['_id']  # Assuming a unique ID for each class
                break  # Stop loop if an active class is found
        except KeyError as e:
            print(f"Missing key: {e}")

    # If no active class is found, return an error message
    if not class_scheduled:
        return jsonify({"error": "No active classes currently available for attendance."})

    # Check if attendance has already been marked for this student and class
    student_id = session.get("student_id")
    if student_id:
        student = mongo.db.registration.find_one({"student_id": student_id})
    attendance_record = mongo.db.ATTENDANCE.find_one({
        'student_id': student_id,
        'class_id': class_id,
        'date': a_date
    })

    if attendance_record:
        return jsonify({"error": "You have already marked attendance."})

    # Load known student images for attendance marking
    path = 'static/StudentImages'
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(os.path.join(path, cl))
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)
    threshold = 0.4
    attendance_marked = False

    while not attendance_marked:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if faceDis[matchIndex] < threshold:
                name = classNames[matchIndex].upper()
                markAttendance(student_id, class_id, name)  # Mark attendance in database
                attendance_marked = True
                break
            else:
                # Draw rectangle and message for unrecognized face
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "You are not my student", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"success": "Attendance marked successfully."})
