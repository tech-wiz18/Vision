from os import system
import cv2
import cv2.aruco as aruco
import numpy as np
import gym
import vision_arena
import pybullet as p

def Image():
    global env

    return env.camera_feed()

def Bot_Pos():
    global A1, A2, aruco_dict, Last_Movement
    while True:
        img = Image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco.DetectorParameters_create())
        try:
            for i in range(len(ids)):
                id = ids[i][0]
                if id == 107:
                    L = corners[i][0]
                    X = int((L[0][0]+L[2][0])/2)
                    Y = int((L[0][1]+L[2][1])/2)

                    MX = int((Y-A1[1])/(A2[1]/9))
                    MY = int((X-A1[0])/(A2[0]/9))

                    X1 = int(((L[0][0]+L[1][0])/2))
                    Y1 = int(((L[0][1]+L[1][1])/2))
                    X2 = int(((L[2][0]+L[3][0])/2))
                    Y2 = int(((L[2][1]+L[3][1])/2))
                    return (X1,Y1),(X2,Y2),(X,Y),(MX,MY)
        except:
            if Last_Movement == "F":
                Reverse()
            elif Last_Movement == "B":
                Forward()

def Detection():
    global A, A1, A2

    color_dict = {'Red'    : np.array([[  0,   0, 145], [  0,   0, 145]], dtype = np.int),
                  'Yellow' : np.array([[  0, 227, 227], [  0, 227, 227]], dtype = np.int)}

    List = [[1,2,3],[4,5,6]]

    for i, color in enumerate(color_dict):
        img = Image()

        blur = cv2.bilateralFilter(img,9,75,75) 
        mask = cv2.inRange(blur, color_dict[color][0] - 10, color_dict[color][1] + 10)
        mask = cv2.dilate(mask, np.ones((5,5), dtype = np.uint8), iterations = 1)
        mask = cv2.erode(mask, np.ones((3,3), dtype = np.uint8), iterations = 1)

        res = cv2.bitwise_and(img, img, mask = mask)

        test = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, test = cv2.threshold(test, 5, 95, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area>100:
                rect = cv2.minAreaRect(contour)
                a=rect[1][0]*rect[1][1]
                m = cv2.moments(contour)
                Y = (m["m10"]/m["m00"])
                X = (m["m01"]/m["m00"])

                y = int((Y-A1[0])/(A2[0]/9))
                x = int((X-A1[1])/(A2[1]/9))

                if (area/a) > 0.9:
                    A[x][y] = List[i][0]
                elif (area/a) <= 0.9 and (area/a) > 0.77:
                    A[x][y] = List[i][1]
                else:
                    A[x][y] = List[i][2]

def Cell(X,Y):
    global Open, Closed
    List = [] 
    
    if [X,Y] == [0,4] and [X+1,Y] not in Open and [X+1,Y] not in Closed:
        List.append([X+1,Y])
    elif [X,Y] == [4,8] and [X,Y-1] not in Open and [X,Y-1] not in Closed:
        List.append([X,Y-1])
    elif [X,Y] == [8,4] and [X-1,Y] not in Open and [X-1,Y] not in Closed:
        List.append([X-1,Y])
    elif [X,Y] == [4,0] and [X,Y+1] not in Open and [X,Y+1] not in Closed:
        List.append([X,Y+1])
    
    if ((X == 0 and Y!=8) or (X == 2 and Y!=0 and Y!=8 and Y!=6)) and [X,Y+1] not in Open and [X,Y+1] not in Closed:
        List.append([X,Y+1])
    elif ((Y == 8 and X!=8) or (Y == 6 and X!=0 and X!=8 and X!=6)) and [X+1,Y] not in Open and [X+1,Y] not in Closed:
        List.append([X+1,Y])
    elif ((X == 8 and Y!=0) or (X == 6 and Y!=0 and Y!=8 and Y!=2)) and [X,Y-1] not in Open and [X,Y-1] not in Closed:
        List.append([X,Y-1])
    elif ((Y == 0 and X!=0) or (Y == 2 and X!=0 and X!=8 and X!=2)) and [X-1,Y] not in Open and [X-1,Y] not in Closed:
        List.append([X-1,Y])
        
    if [X,Y] == [2,4] and [X-1,Y] not in Open and [X-1,Y] not in Closed:
        List.append([X-1,Y])
    elif [X,Y] == [4,6] and [X,Y+1] not in Open and [X,Y+1] not in Closed:
        List.append([X,Y+1])
    elif [X,Y] == [6,4] and [X+1,Y] not in Open and [X+1,Y] not in Closed:
        List.append([X+1,Y])
    elif [X,Y] == [4,2] and [X,Y-1] not in Open and [X,Y-1] not in Closed:
        List.append([X,Y-1])
    
    if [X,Y] == [1,4]:
        if [X+1,Y] not in Open and [X+1,Y] not in Closed:
            List.append([X+1,Y])
        if [X-1,Y] not in Open and [X-1,Y] not in Closed:
            List.append([X-1,Y])
    if [X,Y] == [7,4]:
        if [X-1,Y] not in Open and [X-1,Y] not in Closed:
            List.append([X-1,Y])
        if [X+1,Y] not in Open and [X+1,Y] not in Closed:
            List.append([X+1,Y])
    if [X,Y] == [4,7]:
        if [X,Y-1] not in Open and [X,Y-1] not in Closed:
            List.append([X,Y-1])
        if [X,Y+1] not in Open and [X,Y+1] not in Closed:
            List.append([X,Y+1])
    if [X,Y] == [4,1]:
        if [X,Y+1] not in Open and [X,Y+1] not in Closed:
            List.append([X,Y+1])
        if [X,Y-1] not in Open and [X,Y-1] not in Closed:
            List.append([X,Y-1])

    Open.remove([X,Y])
    Open.extend(List)
    Closed.append([X,Y])
    return List

def End_Path(x,y, Node, f=0):
    global Start, A
    List = []
    if Start == [0,4]:
        for i in range(x+f,4):
            List.append([i,4])
            if A[i][4] == Node:
                if [i,4] == [3,4]:
                    List.append([4,4])
                return List
        List.clear()
        for i in range(x-f,0,-1):
            List.append([i,4])
            if A[i][4] == Node:
                return List
    elif Start == [4,8]:
        for i in range(y-f,4,-1):
            List.append([4,i])
            if A[4][i] == Node:
                if [4,i] == [4,5]:
                    List.append([4,4])
                return List
        List.clear()
        for i in range(y+f,8):
            List.append([4,i])
            if A[4][i] == Node:
                return List
    elif Start == [8,4]:
        for i in range(x-f,4,-1):
            List.append([i,4])
            if A[i][4] == Node:
                if [i,4] == [5,4]:
                    List.append([4,4])
                return List
        List.clear()
        for i in range(x+f,8):
            List.append([i,4])
            if A[i][4] == Node:
                return List
    elif Start == [4,0]:
        for i in range(y+f,4):
            List.append([4,i])
            if A[4][i] == Node:
                if [4,i] == [4,3]:
                    List.append([4,4])
                return List
        List.clear()
        for i in range(y-f,0,-1):
            List.append([4,i])
            if A[4][i] == Node:
                return List
    return None

def Path(Node):
    global A, Paths, End
    path_found = []
    while True:
        i = 0
        size = len(Paths)
        while i<size:
            best_cells = Cell(*Paths[i][-1])
            j = 0
            while j<len(best_cells):
                x,y = best_cells[j]
                if A[x,y] == Node:
                    Paths[i].append([x,y])
                    if len(path_found) != 0:
                        if any([all([path_found[-1] == [4,4],len(Paths[i]) < len(path_found)-1]),len(Paths[i]) < len(path_found)]):
                            return False, Paths[i]
                        else:
                            return True, path_found
                    else:
                        return False, Paths[i]
                if [x,y] in End:
                    Paths[i].append([x,y])
                    List = End_Path(x,y, Node)
                    if List != None:
                        Paths[i].extend(List)
                        path_found = Paths[i]
                    best_cells.remove([x,y])
                    j-=1
                j+=1
                
            if len(best_cells) == 0:
                Paths.remove(Paths[i])
                size-=1
                i-=1
            else:
                Paths[i].append(best_cells[0])
                if len(path_found) != 0:
                    if any([all([path_found[-1] == [4,4],len(Paths[i]) >= len(path_found)-1]),len(Paths[i]) >= len(path_found)]):
                        Paths.remove(Paths[i])
                        i-=1
                best_cells.remove(best_cells[0])
                while len(best_cells) != 0:
                    Paths.append(Paths[i][:-1])
                    Paths[-1].append(best_cells[0])
                    if len(path_found) != 0:
                        if any([all([path_found[-1] == [4,4],len(Paths[-1]) >= len(path_found)-1]),len(Paths[-1]) >= len(path_found)]):
                            Paths.remove(Paths[-1])
                    
                    best_cells.remove(best_cells[0])
            i+=1
            if len(Paths) == 0:
                if len(path_found) == 0:
                    return False, None
                else:
                    return True, path_found

def Distance(a1,b1,a2,b2):
    return ((b2-b1)**2 + (a2-a1)**2)**(1/2)

def Angle(a1,b1,a2,b2):
    bot_vector = complex(a1,b1)
    path_vector = complex(a2,b2)
    angle = np.angle(path_vector/bot_vector)
    angle = (angle*180)/3.14
    return angle

def Forward(d=20):
    global env
    speed = min(10,max(d-12,5))
    env.move_husky(speed,speed,speed,speed)
    for _ in range(min(5,d-10)):
        p.stepSimulation()
    env.move_husky(0,0,0,0)
    p.stepSimulation()

def Reverse(d=20):
    global env
    speed = min(10,max(d-12,5))
    env.move_husky(-speed,-speed,-speed,-speed)
    for _ in range(min(5,d-10)):
        p.stepSimulation()
    env.move_husky(0,0,0,0)
    p.stepSimulation()

def Left(theta):
    global env
    speed = min(23,theta+4)
    env.move_husky(-speed,speed,-speed,speed)
    for _ in range(5):
        p.stepSimulation()
    env.move_husky(0,0,0,0)
    p.stepSimulation()

def Right(theta):
    global env
    speed = min(23,theta+4)
    env.move_husky(speed,-speed,speed,-speed)
    for _ in range(5):
        p.stepSimulation()
    env.move_husky(0,0,0,0)
    p.stepSimulation()

def Move(Best_Path):
    global A1, A2, Last_Movement
    for i,j in Best_Path:
        Y = (i+0.5)*(A2[1]/9) + A1[1]
        X = (j+0.5)*(A2[0]/9) + A1[0]
        while True:
            (X1,Y1), (X2,Y2), (MX,MY), _ = Bot_Pos()
            d = Distance(X,Y,MX,MY)
            if d > 12:
                theta = Angle((X1-X2),(Y1-Y2),(X - MX),(Y - MY))
                if (theta <= 5 and theta >=-5):
                    Last_Movement = "F"
                    Forward(int(d))
                elif theta >= 125 or theta <= -125:
                    if theta >= 175 or theta <= -175:
                        Last_Movement = "B"
                        Reverse(int(d))
                    elif theta >= 125:
                        Left(int(180-theta))
                    elif theta <= -125:
                        Right(int(180-abs(theta)))
                elif theta < 0:
                    Left(-int(theta))
                elif theta > 0:
                    Right(int(theta))
            else:
                break

def init(is_first = False):
    global Interpretation, A, A1, A2, env, aruco_dict, Start, End, Last_Movement, Read
    
    if is_first:
        Interpretation = {"Black":0, "SR":1, "CR":2, "TR":3, "SY":4, "CY":5, "TY":6}
        Read = {1:"Red Square", 2:"Red Circle", 3:"Red Triangle", 4:"Yellow Square", 5:"Yellow Circle", 6:"Yellow Triangle"}
        A = np.zeros([9,9], dtype = np.int32) # Arena
        A1 = np.array([ 18,  18], dtype = np.int32) # Black Stripes Thickness
        A2 = np.array([476, 475], dtype = np.int32) # Image Size

        env = gym.make("vision_arena-v0")

        system('cls')

        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

        Image()
        Detection()
    else:
        env.remove_car()
        env.respawn_car()
        _ = env.camera_feed()
        
    _, _, _, (X,Y) = Bot_Pos()
    
    Start = [X,Y]
    Last_Movement = None

    if [X,Y] == [0,4]:
        End = [[X,Y-1],[X+2,Y-1]]
    elif [X,Y] == [4,8]:
        End = [[X-1,Y],[X-1,Y-2]]
    elif [X,Y] == [8,4]:
        End = [[X,Y+1],[X-2,Y+1]]
    elif [X,Y] == [4,0]:
        End = [[X+1,Y],[X+1,Y+2]]
    
    print("Start",Start)

def End_Run():
    global Interpretation, env, Read
    while True:
        Node = Interpretation[env.roll_dice()]
        _, _, _,(X,Y) = Bot_Pos()
        best_path = End_Path(X,Y, Node, 1)
        if best_path != None:
            print(Read[Node], best_path)
            Move(best_path)
            if best_path[-1] == [4,4]:
                return None
        else:
            print(Read[Node],None)

def Run():
    global Interpretation, env, Open, Closed, Paths, Read
    while True:
        Node = Interpretation[env.roll_dice()]
        _, _, _,(X,Y) = Bot_Pos()
        if [X,Y] not in End:
            Open = [[X,Y]]
            Closed = []
            Paths = [[[X,Y]]]
            check, best_path = Path(Node)
            if best_path != None and not check:
                print(Read[Node], best_path[1:])
                Move(best_path[1:])
            elif check:
                print(Read[Node], best_path[1:])
                Move(best_path[1:])
                if best_path[-1] == [4,4]:
                    print("End",[4,4])
                    return None
                End_Run()
                print("End",[4,4])
                return None
            else:
                print(Read[Node],None)
        else:
            best_path = End_Path(X,Y,Node)
            if best_path!= None:
                print(Read[Node], best_path)
                Move(best_path)
                if best_path[-1] == [4,4]:
                    print("End",[4,4])
                    return None
                End_Run()
                print("End",[4,4])
                return None
            else:
                print(Read[Node],None)