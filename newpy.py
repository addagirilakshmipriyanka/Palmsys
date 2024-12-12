# Initialize the pre-trained MobileNetV2 model for palm feature extraction
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

def preprocess_image(img):
    """
    Preprocess the captured or loaded image for feature extraction.
    """
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def extract_features(img):
    """
    Extract features using MobileNetV2 for the given image.
    """
    features = feature_extractor.predict(img)
    return features.flatten()

def load_encoded_palms(storage_path="static/StudentImages"):
    """
    Load and encode all palm images from the given directory.
    """
    images = []
    classNames = []
    myList = os.listdir(storage_path)
    for cl in myList:
        curImg = cv2.imread(os.path.join(storage_path, cl))
        if curImg is not None:
            images.append(curImg)
            classNames.append(os.path.splitext(cl)[0])
    
    encodeListKnown = []
    for img in images:
        processed_img = preprocess_image(img)
        features = extract_features(processed_img)
        encodeListKnown.append(features)
    
    print('Palm Encoding Complete')
    return encodeListKnown, classNames

def capture_palm():
    """
    Capture palm image using a webcam.
    """
    cap = cv2.VideoCapture(0)
    print("Press 'c' to capture the palm image.")
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to capture image")
            break
        cv2.imshow("Palm Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            palm_image = frame
            break

    cap.release()
    cv2.destroyAllWindows()
    return palm_image

def authenticate_palm(captured_features, encodeListKnown, classNames, threshold=0.9):
    """
    Authenticate captured palm image against known encodings.
    """
    for i, stored_features in enumerate(encodeListKnown):
        similarity = cosine_similarity([captured_features], [stored_features])[0][0]
        print(f"Similarity with {classNames[i]}: {similarity:.4f}")
        if similarity > threshold:
            print(f"Authenticated as {classNames[i]}")
            return classNames[i]
    print("Palm not recognized")
    return None

def markAttendance(student_id, class_id, name):
    """
    Placeholder function for marking attendance.
    """
    print(f"Attendance marked for {name} (Student ID: {student_id}, Class ID: {class_id})")

# Load stored palm encodings
encodeListKnown, classNames = load_encoded_palms()

# Capture palm image
palm_image = capture_palm()

# Process and extract features from the captured image
if palm_image is not None:
    preprocessed_palm = preprocess_image(palm_image)
    captured_features = extract_features(preprocessed_palm)

    # Authenticate and mark attendance
    student_id = "001"  # Example student ID
    class_id = "A1"     # Example class ID
    authenticated_user = authenticate_palm(captured_features, encodeListKnown, classNames)
    
    if authenticated_user:
        markAttendance(student_id, class_id, authenticated_user)
    else:
        print("Authentication failed.")
else:
    print("No palm image captured.")