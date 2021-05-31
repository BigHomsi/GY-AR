import cv2
import dlib
from scipy.spatial import distance
from playsound import playsound

DILBAnsPun = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
HOGAlgo = dlib.get_frontal_face_detector()

def räknaEAR(ÖgaPunkt):
	A = distance.euclidean(ÖgaPunkt[1], ÖgaPunkt[5])
	B = distance.euclidean(ÖgaPunkt[2], ÖgaPunkt[4])
	C = distance.euclidean(ÖgaPunkt[0], ÖgaPunkt[3])
	resul = (A+B)/(2.0*C)
	return resul


Kamera = cv2.VideoCapture(0)


while True:
    _, frame = Kamera.read()
    Grå = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    Ansikter = HOGAlgo(Grå)
    for Ansiktet in Ansikter:

        AnsiPunkter = DILBAnsPun(Grå, Ansiktet)
        VänÖg = []
        HögÖg = []

        for n in range(36,42):
        	x = AnsiPunkter.part(n).x
        	y = AnsiPunkter.part(n).y
        	VänÖg.append((x,y))
        	NäsPun = n+1
        	if n == 41:
        		NäsPun = 36
        	x2 = AnsiPunkter.part(NäsPun).x
        	y2 = AnsiPunkter.part(NäsPun).y
        	cv2.circle(frame,(x,y),1,(0,255,255),1)

        for n in range(42,48):
        	x = AnsiPunkter.part(n).x
        	y = AnsiPunkter.part(n).y
        	HögÖg.append((x,y))
        	NäsPun = n+1
        	if n == 47:
        		NäsPun = 42
        	x2 = AnsiPunkter.part(NäsPun).x
        	y2 = AnsiPunkter.part(NäsPun).y
        	cv2.circle(frame,(x,y),1,(0,255,255),1)

        Vänstra = räknaEAR(VänÖg)
        Högra = räknaEAR(HögÖg)

        EAR = (Vänstra+Högra)/2
        EAR = round(EAR,2)
        if EAR<0.25:
        	playsound('ljud1.mp3')
        	
        print(EAR)

    cv2.imshow("GY arbete", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
Kamera.release()
cv2.destroyAllWindows()