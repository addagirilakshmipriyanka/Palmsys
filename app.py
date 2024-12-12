from flask import Flask, render_template, request, redirect, url_for, jsonify,session,flash
from flask_pymongo import PyMongo
import os
import shutil
from datetime import datetime
import face_recognition
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI applications
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, static_url_path='/static')
app.secret_key = 'kalam@123' 
# Configure MongoDB
app.config['MONGO_URI'] = 'mongodb://localhost:27017/Palmloc'  # Update with your MongoDB URI
mongo = PyMongo(app)


def parse_time(time_str):
    """Try parsing the time string with different formats."""
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(time_str, fmt).time()
        except ValueError:
            continue
    raise ValueError(f"Time data '{time_str}' does not match expected formats")
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encode = encodings[0]
            encodeList.append(encode)
    return encodeList

'''def markAttendance(name):
    mongo.db.Attendance.insert_one({
                    'student_name': name,
                    'timestamp': datetime.now()

                })'''
def markAttendance(student_id, class_id, name):
    mongo.db.ATTENDANCE.insert_one({
        'student_id': student_id,
        'class_id': class_id,
        'date': datetime.today().strftime("%d-%m-%Y"),
        'marked_time': datetime.now().strftime("%H:%M:%S"),
        'status': 'present'
    })
def create_pie_chart(attendance_percentage):
    # Prepare data for the pie chart
    labels = list(attendance_percentage.keys())
    sizes = list(attendance_percentage.values())

    # Clear existing images in static/images directory
    image_path = 'static/images'
    for filename in os.listdir(image_path):
        file_path = os.path.join(image_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    # Create a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Save the pie chart as an image
    plt.savefig(os.path.join(image_path, 'attendance_pie_chart.png'))
    plt.close()

@app.route('/')
def home():
    return render_template('home_page.html')

@app.route('/student')
def student():
    return render_template('stu_login.html', message='')

@app.route('/faculty')
def faculty():
    return render_template('fac_login.html', message='')

@app.route('/admin')
def admin():
    return render_template('admin_login.html', message='')

@app.route('/student_login', methods=['POST'])
def student_login():
    id_no = request.form['id_no']
    password = request.form['pass']
    session['student_id'] = id_no
    # Check credentials in MongoDB
    user = mongo.db.registration.find_one({'student_id': id_no})

    if user:
        return jsonify({'redirect': url_for('student_web')})  # Use URL for the redirect
    else:
        return jsonify({'error': 'Invalid ID or password for Student.'})

@app.route('/student_web')
def student_web():
    return render_template('student/stu_web.html')


@app.route('/student_checkattendance')
def student_checkattendance():
    student_id = session.get('student_id')
    if not student_id:
        return "You need to log in first."

    # Get today's date in the required format
    has_attendance_data = False  # Default flag to False
    pie_chart_path = None  # Initialize the pie chart path
    attendance_data = []  # Initialize attendance data list


        # Fetch all attendance records for the student
    all_attendance = list(mongo.db.ATTENDANCE.find({"student_id": student_id}))

        # Debugging statement to check attendance records
    print("All Attendance Records:", all_attendance)

    subject_counts = {}  # Dictionary to count 'present' attendance per subject
    total_classes = {}  # Dictionary to fetch total classes per subject

        # Count attendance for each subject
    for record in all_attendance:
        subject = mongo.db.Class.find_one({"_id": record['class_id']})
        if subject:
            subject_name = subject['S_Name']
            subject_counts[subject_name] = subject_counts.get(subject_name, 0) + (1 if record['status'] == 'present' else 0)

        # Fetch total classes from Total_Classes collection
    total_classes_data = mongo.db.Total_Classes.find()
    for entry in total_classes_data:
        subject_name = entry['S_Name']
        total_classes[subject_name] = entry['count']

        # Generate pie chart if there are subject counts
    if subject_counts:
        has_attendance_data = True  # Set the flag to True
        labels = list(subject_counts.keys())
        sizes = []

        for subject in labels:
            if total_classes.get(subject, 0) > 0:
                sizes.append((subject_counts[subject] / total_classes[subject]) * 100)
            else:
                sizes.append(0)  # Avoid ZeroDivisionError by setting to 0 if no classes

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', normalize=False)
        pie_chart_path = os.path.join('static', 'images', 'attendance_pie_chart.png')
        plt.savefig(pie_chart_path)
        plt.close()

        # Prepare today's attendance data for rendering (optional: for today's date only)
        today_date = datetime.now().strftime('%d-%m-%Y')
        today_attendance = [
            {
                'class_id': mongo.db.Class.find_one({"_id": record['class_id']})['S_Name'],
                'status': record['status']
            } for record in all_attendance if record['date'] == today_date
        ]

        return render_template(
            'student/check_attendence.html',
            today_attendance=today_attendance,
            student_id=student_id,
            subject_counts=subject_counts,
            total_classes=total_classes,
            has_attendance_data=has_attendance_data,  # Pass the flag
            pie_chart_path=pie_chart_path  # Pass the path for the pie chart
        )
    else:
        # Render with empty data if GET request
        return render_template("student/check_attendence.html", has_attendance_data=False, today_attendance=[], student_id=None)

@app.route('/student_help')
def student_help():
    return render_template('student/help.html')
@app.route('/help', methods=['GET', 'POST'])
def help():
    if request.method == 'POST':
        id_no = request.form['id_no']
        name = request.form['name']
        ph_no = request.form['ph_no']
        email = request.form['email']
        comment = request.form['comment']

        # Insert data into MongoDB
        mongo.db.Help.insert_one({
            "Id_no": id_no,
            "Name": name,
            "phone_no": ph_no,
            "email": email,
            "comment": comment
        })

        # If help_query is successful, return a success message
        return jsonify({"message": "Your query has been sent to HOD"}),200
    return jsonify({"message": "Your query has not been sent to HOD"})  # Render the help page for GET requests

@app.route('/student_personaldetails')
def student_personaldetails():
    student_id = session.get('student_id') 
    print(student_id) # Retrieve logged-in student ID from session
    if student_id:
        student = mongo.db.registration.find_one({"student_id": student_id})  # Fetch details from MongoDB
        return render_template('student/personal_details.html', student=student)
    else:
        return "Student not found", 404

@app.route('/student_takeattendance')
def student_takeattendance():
    return render_template('student/take_attendance.html')

@app.route('/stu_attendance', methods=['POST'])
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
                cv2.putText(img, f"Attendance marked for {name}", cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                break
            else:
                # Draw rectangle and message for unrecognized face
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, "Palm not recognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"success": "Attendance marked successfully."})
    '''
    # Load palm image templates for students
    path = 'static/PalmImages'
    images = []
    classNames = []
    myList = os.listdir(path)
    for cl in myList:
        curImg = cv2.imread(os.path.join(path, cl), cv2.IMREAD_GRAYSCALE)
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

    print('Palm Templates Loaded')
    
    # ORB feature-based matching
    orb = cv2.ORB_create(nfeatures=1000)
    template_features = [(orb.detectAndCompute(img, None)) for img in images]
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Matching function
    def matchPalm(image, template_features):
        keypoints1, descriptors1 = orb.detectAndCompute(image, None)
        best_match_index = -1
        max_good_matches = 0
        for idx, (keypoints2, descriptors2) in enumerate(template_features):
            if descriptors2 is not None:
                matches = bf.match(descriptors1, descriptors2)
                good_matches = [m for m in matches if m.distance < 32]
                if len(good_matches) > max_good_matches and len(good_matches) > 15:
                    max_good_matches = len(good_matches)
                    best_match_index = idx

        return best_match_index
    
    # Capture palm image and match with templates
    cap = cv2.VideoCapture(0)
    attendance_marked = False

    while not attendance_marked:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (400, 400))

        # Match the captured palm with stored templates
        match_index = matchPalm(img_resized, template_features)

        if match_index != -1:
            name = classNames[match_index].upper()
            markAttendance(student_id, class_id, name)
            attendance_marked = True
            cv2.putText(img, f"Attendance marked for {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(img, "Palm not recognized", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Palm Authentication', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({"success": "Attendance marked successfully."})'''


@app.route('/student_upcomingclass')
def student_upcomingclass():
    today = datetime.today().strftime("%d-%m-%Y")
    current_time = datetime.now().time()

    # Query MongoDB for upcoming classes (only those whose end time is in the future)
    upcoming_classes = mongo.db.Class.find({
        'date': today,
        'end_time': {'$gte': current_time.strftime("%H:%M")}
    })

    # Pass upcoming classes to the template
    return render_template('student/upcoming_class.html', classes=upcoming_classes)

# Faculty Routes
@app.route('/faculty_login', methods=['POST'])
def faculty_login():
    id_no = request.form['id_no']
    password = request.form['pass']
    
    # Check credentials in MongoDB
    user = mongo.db.FacultyLogin.find_one({'ID': id_no, 'pass': password})

    if user:
        return jsonify({'redirect': url_for('faculty_web')})
    else:
        return jsonify({'error': 'Invalid ID or password for Faculty.'})

@app.route('/faculty_web')
def faculty_web():
    return render_template('faculty/fac_web.html')

@app.route('/faculty_addclass')
def faculty_addclass():
    return render_template('faculty/add_class.html')

@app.route('/add_class', methods=['POST'])
def add_class():
    # Retrieve form data
    f_name = request.form.get('fac_name')
    s_name = request.form.get('sub_name')
    s_time = request.form.get('start_time')
    e_time = request.form.get('end_time')
    c_date = datetime.today().date().strftime("%d-%m-%Y")
    
    # Checking if there are any classes scheduled
    scheduled_classes = mongo.db.Class.find({}, {'start_time': 1, 'end_time': 1, 'date': 1, '_id': 0})
    
    # Parse submitted times
    parsed_stime = parse_time(s_time)
    parsed_etime = parse_time(e_time)
    print(parsed_stime)
    print(parsed_etime)
    # Check for scheduling conflicts
    for cls in scheduled_classes:
        try:
            # Parse times and date from the database entry
            db_start_time = parse_time(cls['start_time'])
            db_end_time = parse_time(cls['end_time'])
            db_date = cls['date']
            
            # Check for overlapping class at the same time on the same date
            if db_date == c_date and (parsed_stime == db_start_time and parsed_etime == db_end_time):
                
                #return jsonify({"Class": "Some other class is scheduled during this time"})
                flash("Some other class is scheduled during this time.", "error")
                return redirect(url_for('faculty_addclass'))
                
        except KeyError as e:
            print(f"Missing key in class data: {e}")

    # Insert class details into the Class collection
    mongo.db.Class.insert_one({
        'F_Name': f_name,
        'S_Name': s_name,
        'start_time': s_time,
        'end_time': e_time,
        'date': c_date
    })
    
    # Update the Total_Classes count
    total_classes = mongo.db.Total_Classes.find_one({"S_Name": s_name}, {"count": 1, "_id": 0})
    count_ = total_classes['count'] + 1
    mongo.db.Total_Classes.update_one({"S_Name": s_name}, {"$set": {"count": count_}})
    
    return redirect(url_for('faculty_upcomingclass'))

@app.route('/faculty_upcomingclass')
def faculty_upcomingclass():
    today = datetime.today().strftime("%d-%m-%Y")
    current_time = datetime.now().time()

    # Query MongoDB for upcoming classes (only those whose end time is in the future)
    upcoming_classes = mongo.db.Class.find({
        'date': today,
        'end_time': {'$gte': current_time.strftime("%H:%M")}
    })

    # Pass upcoming classes to the template
    return render_template('faculty/upcoming_class.html', classes=upcoming_classes)
@app.route('/faculty_help')
def faculty_help():
    return render_template('faculty/help.html')
# Admin Routes
@app.route('/admin_login', methods=['POST'])
def admin_login():
    id_no = request.form['id_no']
    password = request.form['pass']
    
    # Check credentials in MongoDB
    user = mongo.db.AdminLogin.find_one({'id': id_no, 'password': password})

    if user:
        return jsonify({'redirect': url_for('admin_web')})
    else:
        return jsonify({'error': 'Invalid ID or password for Admin.'})

@app.route('/admin_web')
def admin_web():
    return render_template('admin/admin_web.html')

@app.route('/admin_addclass')
def admin_addclass():
    return render_template('admin/add_class.html')

@app.route('/add_class1', methods=['POST'])
def add_class1():
    # Retrieve form data
    f_name = request.form.get('fac_name')
    s_name = request.form.get('sub_name')
    s_time = request.form.get('start_time')
    e_time = request.form.get('end_time')
    c_date=datetime.today().date().strftime("%d-%m-%Y")
    # Insert class details into the Class collection
    mongo.db.Class.insert_one({
        'F_Name': f_name,
        'S_Name': s_name,
        'start_time':s_time,
        'end_time':e_time,
        'date':c_date
    })
    
    return redirect(url_for('admin_upcomingclass'))


@app.route('/admin_checkattendence')
def admin_checkattendance():
    return render_template('admin/check_attendence.html')

@app.route('/check_attendance', methods=['GET', 'POST'])
def check_attendance():
    has_attendance_data = False  # Default flag to False
    pie_chart_path = None  # Initialize the pie chart path
    attendance_data = []  # Initialize attendance data list

    if request.method == 'POST':
        student_id = request.form.get('id')

        # Fetch all attendance records for the student
        all_attendance = list(mongo.db.ATTENDANCE.find({"student_id": student_id}))

        # Debugging statement to check attendance records
        print("All Attendance Records:", all_attendance)

        subject_counts = {}  # Dictionary to count 'present' attendance per subject
        total_classes = {}  # Dictionary to fetch total classes per subject

        # Count attendance for each subject
        for record in all_attendance:
            subject = mongo.db.Class.find_one({"_id": record['class_id']})
            if subject:
                subject_name = subject['S_Name']
                subject_counts[subject_name] = subject_counts.get(subject_name, 0) + (1 if record['status'] == 'present' else 0)

        # Fetch total classes from Total_Classes collection
        total_classes_data = mongo.db.Total_Classes.find()
        for entry in total_classes_data:
            subject_name = entry['S_Name']
            total_classes[subject_name] = entry['count']

        # Generate pie chart if there are subject counts
        if subject_counts:
            has_attendance_data = True  # Set the flag to True
            labels = list(subject_counts.keys())
            sizes = []

            for subject in labels:
                if total_classes.get(subject, 0) > 0:
                    sizes.append((subject_counts[subject] / total_classes[subject]) * 100)
                else:
                    sizes.append(0)  # Avoid ZeroDivisionError by setting to 0 if no classes

            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%')
            pie_chart_path = os.path.join('static', 'images', 'attendance_pie_chart.png')
            plt.savefig(pie_chart_path)
            plt.close()

        # Prepare today's attendance data for rendering (optional: for today's date only)
        today_date = datetime.now().strftime('%d-%m-%Y')
        today_attendance = [
            {
                'class_id': mongo.db.Class.find_one({"_id": record['class_id']})['S_Name'],
                'status': record['status']
            } for record in all_attendance if record['date'] == today_date
        ]

        return render_template(
            'admin/check_attendence.html',
            today_attendance=today_attendance,
            student_id=student_id,
            subject_counts=subject_counts,
            total_classes=total_classes,
            has_attendance_data=has_attendance_data,  # Pass the flag
            pie_chart_path=pie_chart_path  # Pass the path for the pie chart
        )
    else:
        # Render with empty data if GET request
        return render_template("admin/check_attendence.html", has_attendance_data=False, today_attendance=[], student_id=None)

@app.route('/admin_queries')
def admin_queries():
    queries_data = mongo.db.Help.find()
    
    # Convert the data to a list of dictionaries for easier handling in Jinja
    queries = []
    for query in queries_data:
        queries.append({
            'name': query.get('Name', 'N/A'),
            'id': query.get('Id_no', 'N/A'),
            'phone': query.get('phone_no', 'N/A'),
            'email': query.get('email', 'N/A'),
            'problem': query.get('comment', 'N/A'),
            'verified': query.get('verified', False)
        })
    
    # Render the template with query data
    return render_template('admin/queries.html', queries=queries)

@app.route('/admin_register')
def admin_register():
    return render_template('admin/register.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Retrieve form data
        registration_data = {
            'first_name': request.form.get('first_name'),
            'middle_name': request.form.get('m_name'),
            'last_name': request.form.get('l_name'),
            'dob': request.form.get('dob'),
            'gender': request.form.get('gender'),
            'student_id': request.form.get('Stu_id'),
            'father_name': request.form.get('f_name'),
            'mother_name': request.form.get('m_name'),
            'street_address': request.form.get('street_address'),
            'city': request.form.get('city'),
            'state': request.form.get('state'),
            'country': request.form.get('country'),
            'zip_code': request.form.get('ZIP_code'),
            'email': request.form.get('email'),
            'phone_number': request.form.get('ph_no'),
            'year': request.form.get('Year'),
            'semester': request.form.get('Semister'),
            'branch': request.form.get('Branch'),
            'section': request.form.get('Section'),
            'hall_ticket_number': request.form.get('ht_no'),
            'photo':request.files['p_photo'].filename  # Handle file upload as needed
        }
        
        # Insert data into MongoDB "registration" document
        mongo.db.registration.insert_one(registration_data)
        
        # Handle image copying
        destination_folder = 'static/StudentImages'
        os.makedirs(destination_folder, exist_ok=True)
        image_path = registration_data['photo']
        print(f"Image path provided: {image_path}")

        if os.path.exists(image_path):
            filename = os.path.basename(image_path)
            destination_path = os.path.join(destination_folder, filename)
            shutil.copy(image_path, destination_path)
            print(f"Copied {image_path} to {destination_path}")
        else:
            print(f"Image not found: {image_path}")
        
        return jsonify({"success": "Registration successful!"})
    
    return jsonify({"not success:Registration not successfull!"})

@app.route('/admin_upcomingclass')
def admin_upcomingclass():
    today = datetime.today().strftime("%d-%m-%Y")
    current_time = datetime.now().time()

    # Query MongoDB for upcoming classes (only those whose end time is in the future)
    upcoming_classes = mongo.db.Class.find({
        'date': today,
        'end_time': {'$gte': current_time.strftime("%H:%M")}
    })

    # Pass upcoming classes to the template
    return render_template('admin/upcoming_class.html', classes=upcoming_classes)

if __name__ == '__main__':
    app.run(debug=True)
