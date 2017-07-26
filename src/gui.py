import scipy.io as sio
import numpy as np
import pygame.font, pygame.event, pygame.draw
import pygame
import scipy.optimize as sc
from pygame.locals import *
import heapq
import random

changed = False
counter = 0
Xtrain, Xtest, ytrain, ytest = [], [], [], []
num_hidden = 25
num_input = 900
num_lables = 10
screen = None

def splitData(X, y):
    """Split sample data into training set (80%) and testing set (20%)"""
    
    size1 = X.shape[0] * 0.8
    size2 = X.shape[0] * 0.2
    Xtrain = np.zeros((size1,X.shape[1]))
    Xtest = np.zeros((size2,X.shape[1]))
    ytrain = np.zeros((size1,1))
    ytest = np.zeros((size2,1))
    for i, v in enumerate(np.random.permutation(len(y))):
        #print(i, y[v], len(X[v]))
        
        try:
            Xtrain[i] = X[v]
            ytrain[i] = y[v]
        except:
            Xtest[i-size1] = X[v]
            ytest[i-size1] = y[v]
    return Xtrain, Xtest, ytrain, ytest
    
def calculateImage(background, screen, Theta1, Theta2, lineWidth):
    """Crop and resize the input"""
    
    global changed
    focusSurface = pygame.surfarray.array3d(background)
    focus = abs(1-focusSurface/255)
    focus = np.mean(focus, 2) 
    x = []
    xaxis = np.sum(focus, axis=1)
    for i, v in enumerate(xaxis):
        if v > 0:
            x.append(i)
            break
    for i, v in enumerate(xaxis[ : :-1]):
        if v > 0:
            x.append(len(xaxis)-i)
            break
    
    y = []
    yaxis = np.sum(focus, axis=0)
    for i, v in enumerate(yaxis):
        if v > 0:
            y.append(i)
            break
    for i, v in enumerate(yaxis[ : :-1]):
        if v > 0:
            y.append(len(yaxis)-i)
            break

    try:
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        bound = focus.shape[0]      
        if dx > dy:
            d = dx-dy
            y0t = y[0] - d//2
            y1t = y[1] + d//2+d%2
            if y0t < 0: y0t = y[0]; y1t = y[1] + d
            if y1t > bound: y0t = y[0] - d; y1t = y[1]
            y[0], y[1] = y0t, y1t
        else:
            d = dy-dx
            x0t = x[0] - d//2
            x1t = x[1] + d//2+d%2
            if x0t < 0: x0t = x[0]; x1t = x[1] + d
            if x1t > bound: x0t = x[0] - d; x1t = x[1]
            x[0], x[1] = x0t, x1t 
        dx = x[1]-x[0]
        dy = y[1]-y[0]
        changed = True
        crop_surf =  pygame.Surface((dx,dy))
        crop_surf.blit(background,(0,0),(x[0],y[0],x[1],y[1]), special_flags=BLEND_RGBA_MAX)
        scaledBackground = pygame.transform.smoothscale(crop_surf, (30, 30))
            
        image = pygame.surfarray.array3d(scaledBackground)
        image = abs(1-image/253)
        image = np.mean(image, 2) 
        image = np.matrix(image.ravel())
        (value, prob), (value2, prob2) = probabilty(Theta1,Theta2,image)
        prob = round(prob,1)
        prob2 = round(prob2, 1)
                   
        return [value,prob]

    except:
        image = np.zeros((30,30))

    return False, False
    
def sigmoid(z):
    """Calculate sigmoid function"""
    
    return 1/(1+np.power(np.e,-z))

def probabilty(Theta1, Theta2, X):
    """Classify number and caluclate probability"""
    
    X = np.append(np.ones(shape=(X.shape[0],1)),X,axis=1)
    input = Theta1*np.matrix(X.transpose())
    hiddenLayer = sigmoid(input)
    hiddenLayer = np.append(np.ones(shape=(1,hiddenLayer.shape[1])),hiddenLayer,axis=0)
    proba = sigmoid(Theta2*hiddenLayer)
    l0 = np.array(proba.ravel())[0]
    l1 = heapq.nlargest(2, l0)
    proba2 = l1[1]
    estimate2 = int(np.where(l0==l1[1])[0]+1)
    estimate2 = estimate2 if estimate2<10 else 0
    number = int(proba.argmax(0).transpose())
    estimate = number+1 if number<9 else 0

    return (estimate, float(proba[number])*100), (estimate2, proba2*100)

    
def checkKeys(myData):
    """Detect various keyboard inputs"""
    
    (event, background, drawColor, lineWidth, keepGoing, screen, image) = myData
    
    if event.key == pygame.K_q:
        keepGoing = False
    elif event.key == pygame.K_c:
        clear(background);

    myData = (event, background, drawColor, lineWidth, keepGoing)
    return myData

def clear(background):
    background.fill((255, 255, 255))
    return True

def showStats(value, prob, value2, prob2):
    """ shows the current statistics """
    
    myFont = pygame.font.SysFont("Verdana", 50)
    text = myFont.render("Estimate:    %s" % (value), 1, ((255, 255, 255)))
    screen.blit(text, (10, 370))

    proba = "Probability: %s" % (prob)
    text = myFont.render(proba+"%", 1, ((255, 255, 255)))
    screen.blit(text, (10, 420))

    myFont = pygame.font.SysFont("Verdana", 25)
    text = myFont.render("Second estimate:    %s" % (value2), 1, ((255, 255, 255)))
    screen.blit(text, (10, 490))

    proba = "Probability: %s" % (prob2)
    text = myFont.render(proba+"%", 1, ((255, 255, 255)))
    screen.blit(text, (10, 515))

    return True

def sigmoidGradient(z):
    """Gradient of sigmoid function"""
    
    return np.multiply(sigmoid(z),(1-sigmoid(z)));


def vectorize(v1, v2):
    """Merge to vectors together"""
    
    return np.append(np.ravel(v1), np.ravel(v2))


def backProp(p, num_input, num_hidden, num_labels, X, yvalue, l=0.2):
    """Backpropagation algorithm of neural network"""
    
    Theta1 = np.reshape(p[:num_hidden*(num_input+1)], (num_hidden,-1))
    Theta2 = np.reshape(p[num_hidden*(num_input+1):], (num_labels,-1))
    m = len(X)
    delta1 = 0
    delta2 = 0
    for t in range(m):

        a1 = np.matrix(np.append([1],X[t],axis=0)).transpose()
        z2 = Theta1*a1
        a2 = np.append(np.ones(shape=(1,z2.shape[1])), sigmoid(z2),axis=0)
        z3 = Theta2*a2
        a3 = sigmoid(z3)
        w = np.zeros((num_labels,1))
        w[int(yvalue[t])-1] = 1
        d3 = (a3-w)
        d2 = np.multiply(Theta2[:,1:].transpose()*d3, sigmoidGradient(z2))
        delta1 += d2*a1.transpose()
        delta2 += d3*a2.transpose()
        
    
    Theta1_grad = (1/m)*delta1 + (l/m)*np.append(np.zeros(shape=(Theta1.shape[0],1)), Theta1[:,1:], axis=1);
    Theta2_grad = (1/m)*delta2 + (l/m)*np.append(np.zeros(shape=(Theta2.shape[0],1)), Theta2[:,1:], axis=1);
    answer = vectorize(Theta1_grad, Theta2_grad)
    return answer


def button(x,y,w,h,ic,ac,background,action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()
    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(screen, ac,(x,y,w,h))

        if click[0] == 1 and action != None:
            if(action == "clear"):
                clear(background)
            elif(action == "quit"):
                pygame.quit()
                keepGoing = False
            elif (action == "play2"):
                play2(background)
  
    else:
        pygame.draw.rect(screen, ic,(x,y,w,h))

    
def text_objects(text, font):
    black = (0, 0, 0)
    textSurface = font.render(text, True, black)
    return textSurface,  textSurface.get_rect()

def play():

    r = random.randint(0,9)
    music = "assets/%d.wav" %(r,)
    pygame.mixer.music.load(music)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy()==True:
        continue
    return r

def play2(number):
    music = "assets/%d.wav" %(number,)
    pygame.mixer.music.load(music)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy()==True:
        continue

def wrong():
    return True

def main():
    """Main method. Draw interface"""
    
    global screen
    pygame.init()
    screen = pygame.display.set_mode((900, 800)) #largeur hauteur
    pygame.display.set_caption("Learn by playing with Foxy !")
    screen.fill((246, 245, 242))
    
    header = pygame.Surface((900,50))
    header.fill((249, 105, 14)) #couleur gauche

    border = pygame.Surface((360,360))
    border.fill((249, 105, 14)) #couleur gauche
    background = pygame.Surface((340,340))
    background.fill((255, 255, 255)) #couleur gauche

    background2 = pygame.Surface((360,360))

    screen.blit(header, (0, 0))
    screen.blit(border, (20, 70))
    screen.blit(background, (30, 80))

    clock = pygame.time.Clock()
    keepGoing = True
    lineStart = (0, 0)
    drawColor = (0, 0, 0)
    lineWidth = 10
    
    inputTheta = sio.loadmat('scaledTheta.mat')
    theta = inputTheta['t']
    num_hidden = 25
    num_input = 900
    num_lables = 10

    Theta1 = np.reshape(theta[:num_hidden*(num_input+1)], (num_hidden,-1))
    Theta2 = np.reshape(theta[num_hidden*(num_input+1):], (num_lables,-1))

    pygame.display.update()
    image = None

    #play()
    number = play()

    right_img = pygame.image.load('assets/right.png')
    wrong_img = pygame.image.load('assets/wrong.png')

    fox_happy = pygame.image.load('assets/fox_happy.png')
    fox_sad = pygame.image.load('assets/fox_sad.png')
    fox = pygame.image.load('assets/fox.png')

    chat = pygame.image.load('assets/chat.png')

    myFont = pygame.font.SysFont("Verdana", 25)
    text1 = myFont.render("Use the white area", 1, ((0, 0, 0)))
    text2 = myFont.render("to draw the number", 1, ((0, 0, 0)))
    text3 = myFont.render("you hear !", 1, ((0, 0, 0)))
    screen.blit(text1, (495, 550))
    screen.blit(text2, (495, 580))
    screen.blit(text3, (555, 610))

    screen.blit(chat,(470,500))
    screen.blit(fox,(0,450))

            
    while keepGoing:
        
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                keepGoing = False
            elif event.type == pygame.MOUSEMOTION:
                lineEnd = pygame.mouse.get_pos()
                if pygame.mouse.get_pressed() == (1, 0, 0):
                    x1, y1 = lineStart
                    x2, y2 = lineEnd
                    start = (x1-30, y1-80)
                    end = (x2-30, y2-80)
                    pygame.draw.line(background, drawColor, start, end, lineWidth)
                lineStart = lineEnd
            elif event.type == pygame.MOUSEBUTTONUP:
                screen.fill((246, 245, 242))
                screen.blit(fox,(0,450))
                #w = threading.Thread(name='worker', target=worker)
                image,prob = calculateImage(background, screen, Theta1, Theta2, lineWidth)
                
                if image==number:
                    myFont = pygame.font.SysFont("Verdana", 45)
                    text = myFont.render("Well Done !", 1, ((41, 150, 9)))
                    screen.blit(text, (490, 580))

                    screen.blit(chat,(470,500))
                    screen.blit(fox_happy,(0,450))
                    screen.blit(right_img,(405,75))
                    clear(background)
                    number = play()

                elif image!=False :
                    if(prob > 90):  
                        myFont = pygame.font.SysFont("Verdana", 45)
                        text = myFont.render("Try Again !", 1, ((228, 6, 19)))
                        screen.blit(text, (490, 580))


                        screen.blit(chat,(470,500))
                        screen.blit(fox_sad,(0,450))
                        screen.blit(wrong_img,(405,75))
                        clear(background)
                        play2(number) 

            elif event.type == pygame.KEYDOWN:
                myData = (event, background, drawColor, lineWidth, keepGoing, screen, image)
                myData = checkKeys(myData)
                (event, background, drawColor, lineWidth, keepGoing) = myData


        screen.blit(header, (0, 0))
        screen.blit(border, (20, 70))
        screen.blit(background, (30, 80))
        
        eraser_img = pygame.image.load('assets/eraser.png')
        play_img = pygame.image.load('assets/play.png')
        cancel_img = pygame.image.load('assets/cancel.png')
    
        screen.blit(eraser_img,(798,88))
        screen.blit(play_img,(798,218))
        screen.blit(cancel_img,(798,348))

        myFont = pygame.font.SysFont("Verdana", 35)
        text = myFont.render("Learn by playing with Foxy !", 1, ((255, 255, 255)))
        screen.blit(text, (250, 5))

        pygame.display.flip()
        button_color = (249, 105, 14)
        button_color_down = (216,84,5)
        button(780,70,100,100,button_color,button_color_down,background,"clear")
        button(780,200,100,100,button_color,button_color_down,number,"play2")
        button(780,330,100,100,button_color,button_color_down,background,"quit")




if __name__ == "__main__":
    main()