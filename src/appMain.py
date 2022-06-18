import sys
from collections import Counter

from PyQt5 import QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt, QPoint, QTimer
from PyQt5.QtGui import QPainter, QPen

from model_loaders import house_loader, tree_loader, person_loader

"""
Main application
"""

# Initialize empty window that will hold the results
res_window = None


class draw_screen(QtWidgets.QMainWindow):
    """
    The main application where the participant draws 3 images for
    a house, a tree, and a person.
    """

    def __init__(self):
        super(draw_screen, self).__init__()
        # Load the UI from file that was designed in Qt Designer
        uic.loadUi("resources/app.ui", self)
        # Change the window title and icon
        self.setWindowIcon(QtGui.QIcon('../resources/tuappicon.png'))
        self.setWindowTitle("CSE3000")
        # Initialize the application phase
        # 1: House drawing, 2: Tree drawing, 3: Person drawing
        self.phase = 1
        # Flag that controls if the participant is able to draw or not
        self.ready = False
        # Flag that indicates if the participant is drawing or not
        self.drawing = False

        # Variable that will hold the results for each drawing
        self.res = [0, 0, 0]

        # Initialize the timer with 60 seconds
        self.timer = QTimer()
        self.timer.timeout.connect(self.display)
        self.count = 45
        self.timer_text.setText(str(45))

        # Initialize the brush size and color for drawing
        # 8px width, black color, 448x448 canvas
        self.brushSize = 8
        self.brushColor = Qt.black
        self.lastPoint = QPoint()
        canvas = QtGui.QPixmap(448, 448)
        canvas.fill(Qt.white)
        self.image.setPixmap(canvas)

        # Initialize progress bar to indicate remaining time
        self.progressBar.setValue(100)
        self.progressBar.setFormat("")

        # Initial instruction text
        self.label.setText("Draw a House")

        # Add functionality to each of the buttons
        # Disable the next phase button until something is drawn
        self.start_timer.clicked.connect(self.countdown)
        self.next.setEnabled(False)
        self.next.clicked.connect(self.nextPhase)
        self.reset.clicked.connect(self.resetTimer)
        self.clear.clicked.connect(self.clearCanvas)

    def mousePressEvent(self, event):
        """
        Method that detects when the mouse is pressed down
        Get the (x,y) coordinates of the cursor when the mouse was pressed
        Translate the (x,y) coordinates by the position of the canvas on screen
        :param event: Mouse clicked down event
        """
        if event.button() == Qt.LeftButton & self.ready:
            self.drawing = True
            self.lastPoint = QPoint(event.pos().x() - 70, event.pos().y() - 60)

    def mouseMoveEvent(self, event):
        """
        Method that detects when the mouse was moved
        :param event: Mouse moved
        """
        # If the user is supposed to draw and is clicking the mouse
        # Draw a line between the last 2 recorded points and update the window
        # Results in a line on the canvas under where the mouse was clicked
        if (event.buttons() & Qt.LeftButton) & self.drawing & self.ready:
            painter = QPainter(self.image.pixmap())
            painter.setPen(QPen(self.brushColor, self.brushSize,
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, QPoint(event.pos().x() - 70, event.pos().y() - 60))
            self.lastPoint = QPoint(event.pos().x() - 70, event.pos().y() - 60)
            self.update()

    def mouseReleaseEvent(self, event):
        """
        Method that detects when the mouse button was released
        :param event: Mouse button released
        """
        if event.button() == Qt.LeftButton & self.ready:
            # Stop the drawing
            self.drawing = False

    def clearCanvas(self):
        """
        Method that clears the canvas completely
        """
        self.image.pixmap().fill(Qt.white)
        self.update()

    def resetTimer(self):
        """
        Method that stops and resets the timer
        back to 60 seconds and clears the canvas
        Disable drawing until the timer is started again
        """
        self.timer.stop()
        self.count = 45
        self.progressBar.setValue(100)
        self.timer_text.setText(str(45))
        self.ready = False
        self.next.setEnabled(False)
        self.clearCanvas()
        self.instruction.setHidden(False)

    def nextPhase(self):
        """
        Save the image for the current phase. 448x448 from drawing to 224x224 for input
        Move to the next phase of drawing and change the drawing instruction
        """
        if self.phase == 1:
            # Save house image and move to phase 2
            self.image.pixmap().scaled(224, 224).save('toPredict/predictHouse.png')
            self.label.setText("Draw a Tree")
        elif self.phase == 2:
            # Save tree image and move to phase 3
            self.image.pixmap().scaled(224, 224).save('toPredict/predictTree.png')
            self.label.setText("Draw a Person")
        elif self.phase == 3:
            # Save person image and start calculating results
            self.label.setText("Loading results...")
            self.image.pixmap().scaled(224, 224).save('toPredict/predictPerson.png')
            self.results()
            global res_window
            # Initialize the results screen
            res_window = res_screen(self.res)
            res_window.show()
            # Close the current window
            self.close()
        else:
            # Should never end up here
            print("Something went really wrong")
        # Move to the next phase for drawing
        self.phase += 1
        # Reset the interface for the next phase
        self.resetTimer()
        self.next.setEnabled(False)

    def results(self):
        """
        Get 3 predictions for each of the 3 images with a differently trained model
        The label that is predicted most for each image is the overall resulting label
        """
        housePredict = [0, 0, 0]
        treePredict = [0, 0, 0]
        personPredict = [0, 0, 0]
        # 3 Predictions for the house image that was drawn
        housePredict[0] = house_loader.predict('model/house/house_model_10.tar', 'toPredict/predictHouse.png')
        housePredict[1] = house_loader.predict('model/house/house_model_12.tar', 'toPredict/predictHouse.png')
        housePredict[2] = house_loader.predict('model/house/house_model_15.tar', 'toPredict/predictHouse.png')

        # 3 Predictions for the tree image that was drawn
        treePredict[0] = tree_loader.predict('model/tree/tree_model_10.tar', 'toPredict/predictTree.png')
        treePredict[1] = tree_loader.predict('model/tree/tree_model_12.tar', 'toPredict/predictTree.png')
        treePredict[2] = tree_loader.predict('model/tree/tree_model_15.tar', 'toPredict/predictTree.png')

        # 3 Predictions for the person image that was drawn
        personPredict[0] = person_loader.predict('model/person/person_model_10.tar', 'toPredict/predictPerson.png')
        personPredict[1] = person_loader.predict('model/person/person_model_12.tar', 'toPredict/predictPerson.png')
        personPredict[2] = person_loader.predict('model/person/person_model_15.tar', 'toPredict/predictPerson.png')

        # Find the most commonly predicted label for each of the images and store it
        self.res[0] = Counter(housePredict).most_common()[0][0]
        self.res[1] = Counter(treePredict).most_common()[0][0]
        self.res[2] = Counter(personPredict).most_common()[0][0]

        # Debug
        # print(self.res)

    def countdown(self):
        """
        Method that starts the timer on a different thread than the one
        that handles the GUI for the rest of the application
        The timer will signal at 1 second intervals
        Enable drawing for the user while the timer is ticking down
        """
        self.instruction.setHidden(True)
        self.next.setEnabled(True)
        self.ready = True
        self.timer.start(1000)

    def display(self):
        """
        Method that updates the progress bar and the text on the label
        according to how much time is left
        """
        self.count -= 1
        if not self.count > 0:
            self.timer.stop()
            self.ready = False
            self.timer_text.setText(str(0))
            self.progressBar.setValue(0)
            return
        self.timer_text.setText(str(self.count))
        norm = int(self.count / 45 * 100)
        self.progressBar.setValue(norm)


class res_screen(QtWidgets.QMainWindow):
    """
    The results screen that shows up after the user has drawn the 3 images
    The screen will contain the outcome for each of the images that were drawn
    :param results: The label that was predicted by the models for each image
    """
    def __init__(self, results):
        super(res_screen, self).__init__()

        # Predicted labels for each image
        self.image_results = results
        # Load the UI that was designed in Qt Designer
        uic.loadUi("resources/res_screen.ui", self)
        # Add functionality to the "Next" button
        self.next_result.clicked.connect(self.next_res)
        # Set the result window title
        self.setWindowTitle("CSE3000")

        # The list of features that the house drawing might indicate
        feature_list_0 = [["Stress", "Anxiety"],
                          ["Low self-esteem", "Withdrawal", "Introversion"],
                          ["High self-esteem", "Fantasizing", "Extroversion"]]
        # The list of features that the tree drawing might indicate
        feature_list_1 = [["Depression", "Low Energy"],
                          ["Introversion", "Low ego-strength"],
                          ["Extroversion", "Ambition", "High ego-strength"]]
        # The list of features that the person drawing might indicate
        feature_list_2 = [["Depression", "Low Energy"],
                          ["Withdrawal", "Lack of motivation", "Boredom"],
                          ["Anxiety", "Obsession"]]

        # The list of messages that will be suggested to the user based on the house drawing
        msg_list_0 = [
            "If your responsibilities are causing you stress, try organizing your tasks. Get a friend and do "
            "something you enjoy. Work on your tasks in small intervals and remember to take time for yourself!",
            #####################################################################################################
            "If you're feeling uncertain with yourself take some time off and do things that make you happy. "
            "Socializing can sometimes be taxing, dedicate some time for yourself, watch a show that you like, "
            "reach out to people that are close to you even if you're far away!",
            #####################################################################################################
            "Feeling comfortable with yourself is a great deal! Socializing is the perfect way to create "
            "memories, meet new people and gain experience. Keep working towards your goals and always take "
            "care of yourself. "
        ]
        # The list of messages that will be suggested to the user based on the tree drawing
        msg_list_1 = [
            "Feeling down and demotivated happens to everyone, especially if you are away from people that make "
            "you feel safe and happy. It is important to allow yourself time and to not give up. A tired mind "
            "can't see clearly, allow yourself time to rest  and take things one step at a time",
            #####################################################################################################
            "Staying motivated is not an easy task, especially if you are faced with problems and challenges. "
            "Devoting time to yourself is key improve your wellbeing. Expose yourself to new  experiences and "
            "reflect on the positive aspects of your  personality.",
            #####################################################################################################
            "Having a clear goal in mind is key to helping yourself stay motivated even in the most dire "
            "situations. Enjoy time with friends, expose yourself to new experiences and keep looking ahead! "
        ]
        # The list of messages that will be suggested to the user based on the person drawing
        msg_list_2 = [
            "Feeling down and demotivated happens to everyone, especially if you are away from people that make "
            "you feel safe and happy. It is important to allow yourself time and to not give up. A tired mind "
            "can't see clearly, allow yourself time to rest and take things one step at a time.",
            #####################################################################################################
            "Withdrawal and/or lack of motivation and/or boredom Finding motivation is not always an easy task. "
            "Keeping yourself on a schedule will help you deal with your responsibilities and give you a sense "
            "of accomplishment. Expose yourself to  new experiences, discover what you enjoy the most and "
            "always dedicate for you!",
            #####################################################################################################
            "Feeling overwhelmed happens to everyone. Dealing with pressure is not always trivial or "
            "straightforward. It is important take breaks and spend time doing activities for your being. "
            "Always try to keep a balance between your obligations and leisure. "
        ]
        # Create a list that contains all the features for all images
        self.all_features = [feature_list_0, feature_list_1, feature_list_2]
        # Create a list that contains all the messages for all images
        self.all_msgs = [msg_list_0, msg_list_1, msg_list_2]

        # Dummy string initialize
        feature_str = ""
        # Array containing color codes to format the successive feature labels with
        colors = ["color: #003049;", "color: #d62828;", "color: #f77f00;"]
        # Build and format the string containing the features for the image that is currently shown
        for i, w in enumerate(self.all_features[0][self.image_results[0]]):
            feature_str += '<b style="{}">{}</b>'.format(colors[i], w)
            if i != len(self.all_features[0][self.image_results[0]]) - 1:
                feature_str += " / "

        self.features.setText(feature_str)
        # Set the message for the image that is currently shown
        self.msg.setText(self.all_msgs[0][self.image_results[0]])
        # Introduction message for the image that is currently shown
        self.intro_msg.setText("Your drawing of a house shows signs of:")
        # Initialize the state with the house state
        # 0: Results for house, 1: Results for tree, 2: Results for person
        self.state = 0

    def next_state(self):
        """
        Method that shows the correct features and messages for the image that is currently shown
        Adjusts the introduction text and resets the screen for the next result to be shown
        :return:
        """
        # Showing results for the tree image
        if self.state == 1:
            feature_str = ""
            colors = ["color: #003049;", "color: #d62828;", "color: #f77f00;"]
            for i, w in enumerate(self.all_features[1][self.image_results[1]]):
                feature_str += '<b style="{}">{}</b>'.format(colors[i], w)
                if i != len(self.all_features[1][self.image_results[1]]) - 1:
                    feature_str += " / "

            self.features.setText(feature_str)
            self.msg.setText(self.all_msgs[1][self.image_results[1]])
            self.intro_msg.setText("Your drawing of a tree shows signs of:")
        # Showing results for the person image
        elif self.state == 2:
            feature_str = ""
            colors = ["color: #003049;", "color: #d62828;", "color: #f77f00;"]
            for i, w in enumerate(self.all_features[2][self.image_results[2]]):
                feature_str += '<b style="{}">{}</b>'.format(colors[i], w)
                if i != len(self.all_features[2][self.image_results[2]]) - 1:
                    feature_str += " / "

            self.features.setText(feature_str)
            self.msg.setText(self.all_msgs[2][self.image_results[2]])
            self.intro_msg.setText("Your drawing of a person shows signs of:")

    def next_res(self):
        """
        Move to the next result
        """
        self.state += 1
        self.next_state()

# Starting code for the entire application
app = QtWidgets.QApplication(sys.argv)
app_icon = QtGui.QIcon('./resources/tuappicon.png')
app.setWindowIcon(app_icon)
draw_window = draw_screen()
draw_window.setWindowIcon(app_icon)
draw_window.show()
sys.exit(app.exec_())
