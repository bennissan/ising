#
#        _==/          i     i          \==_
#      /XX/            |\___/|            \XX\
#    /XXXX\            |XXXXX|            /XXXX\
#   |XXXXXX\_         _XXXXXXX_         _/XXXXXX|
#  XXXXXXXXXXXxxxxxxxXXXX A XXXXxxxxxxxXXXXXXXXXXX
# |XXXX MATHEMATICAL MODEL OF FERROMAGNETISM XXXXX|
# XXXXXXXXXXX IN STATISTICAL MECHANICS XXXXXXXXXXXX
# |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|
#  XXXXXX/^^^^"\XXXXXXXXXXXXXXXXXXXXX/^^^^^\XXXXXX
#   |XXX|       \XXX/^^\XXXXX/^^\XXX/       |XXX|
#     \XX\       \X/    \XXX/    \X/       /XX/
#        "\       "      \X/      "      /"
#
#                     Ising.py
#               
#        Nolan Hawkins, Ben Nissan, Justin Lee
#
#      A tool for the simulation of the Ising Model
#       Using Wolff and Metropolis algorithms on 
#   1D, 2D, and 3D square, and 2D triangular lattices
#
#   If run without command-line parameters, this shows a
#       visualization of the Ising model, with many configurable options
#   Otherwise, it will generate a space-sperated value file
#       The first element in each line will be the temperature
#       The following elements will be absolute magnetization after
#       a specific number of iterations
#       Optional configuration arguments are:
#               -model=wolff or -model=metropolis (default)
#               -lattice=1d, -lattice=2d (default), -lattice=3d, or -lattice=triangle
#               -size=NUMBER (this is the side length of the lattice, default is 50)
#               -iterations=NUMBER (this is the number of iterations per 
#                                       temperature that are tested, by default 500)
#               -outputFreq=NUMBER (a absolute magnetization point will be ouput every
#                                       NUMBER iterations, by default 100)
#       Any other arguments will be ignored

import numpy as np
import Tkinter as tk
import sys

# -------------------- Constants --------------------- #

# side length of square window to display animation in
WINDOW_SIZE = 500

# side length of spin array
SPIN_SIZE = 50

# polygon sizes for Tkinter rendering
BLOCK_SIZE = WINDOW_SIZE / SPIN_SIZE
TRI_SIZE = WINDOW_SIZE/(SPIN_SIZE- + .5)
HEX_SIZE = (1/np.sqrt(3)) * TRI_SIZE

# metropolisis flips per frame
FREQUENCY = SPIN_SIZE ** 2

DIMENSIONS = 2

# ----------- Find Neighbors Functions -------------- #
# findNeighbors Functions returns the coordinates of all directly adjacent 
# neighboring lattice points in a list of tuples.

def findNeighbors_1D(grid, coord):
   left  = ((coord[0] - 1) % grid.shape[0],)
   right = ((coord[0] + 1) % grid.shape[0],)
   return [left, right]

def findNeighbors_2D(grid, coord):
    width  = grid.shape[0]
    height = grid.shape[1]
    x = coord[0]
    y = coord[1]
    left   = ((x - 1) % width, y               )
    right  = ((x + 1) % width, y               )
    top    = ( x             , (y - 1) % height)
    bottom = ( x             , (y + 1) % height)
    return [left, right, top, bottom]
    
def findNeighbors_3D(grid, coord):
    width  = grid.shape[0]
    height = grid.shape[1]
    length = grid.shape[2]
    x = coord[0]
    y = coord[1]
    z = coord[2]
    left   = ((x - 1) % width, y               , z               )
    right  = ((x + 1) % width, y               , z               )
    top    = ( x             , (y - 1) % height, z               )
    bottom = ( x             , (y + 1) % height, z               )
    back   = ( x             , y               , (z - 1) % length)
    front  = ( x             , y               , (z + 1) % length)
    return [left, right, top, bottom, back, front]
    
def findNeighbors_tri(grid, coord):
    width  = grid.shape[0]
    height = grid.shape[1]
    x = coord[0]
    y = coord[1]

    # calculate offset for triangle lattice 
    # This is because we are reprsenting the hexagon lattice
    # as a "zig-zag" pressed square lattice. Lining up each
    # horizontal lattice line in an alternating manner.
    offset=1
    if(y%2 == 0):
        offset=0

    NE = ((x+offset)   % width, (y-1) % height)
    NW = ((x+offset-1) % width, (y-1) % height)
    W  = ((x-1)        % width, y             )
    SW = ((x+offset-1) % width, (y+1) % height)
    SE = ((x+offset)   % width, (y+1) % height)
    E  = ((x+1)        % width, y             )
    return [NE, NW, W, SW, SE, E]
    
# ----------- Ising Model Monte Carlo Algorithms -------------- #

# Liklihood for a singular spot will flip based on neighbors and 
# exchange parameter. 
def energyDiff(spins, coord, findNeighbors, J):
    neighbors=findNeighbors(spins, coord)
    s = 0
    for c in neighbors:
        s += spins.item(c)
    return 2*J*spins.item(coord)*s

# Metropolisis single flip algorithm
def metropolis(spins, temp, j, findNeighbors):
    size = spins.shape[0]
    for i in range(FREQUENCY):
        # random sampling
        coord=tuple([np.random.random_integers(0, size-1) for i in spins.shape])
        
        deltaE = energyDiff(spins, coord, findNeighbors, j) 
        #This is the basically the entire metropolis algorithm
        if(deltaE < 0 or np.random.random() < np.exp( - deltaE / temp)):
                spins.itemset(coord, -1 * spins.item(coord))

# Cluster flip algorithm
def wolff(spins, temp, j, findNeighbors):
    size = spins.shape[0]
    # add probability of the cluster
    probability = 1-np.exp(-2*j*(1.0/(temp)))
    
    # A stack based cluster forming algorithm, 10 cluster flip per frame
    for i in range(1):
        # random sampling
        coord=tuple([np.random.random_integers(0, size-1) for i in spins.shape])

        neighbors = findNeighbors(spins, coord)
        localSpin = spins.item(coord)
        spins.itemset(coord, -localSpin)

        for n in neighbors:
            if(spins.item(n) == localSpin and np.random.random() < probability):
                spins.itemset(n, -localSpin)
                neighbors += findNeighbors(spins, n)


# Average spin throughout lattice, generalized for n-dimensions using nditer
def magnetization(spins):
    mags = 0
    for i in np.nditer(spins):
        mags += i
    mags /= spins.size
    return mags

#Return a n-cube lattice of dimension dimensions with a side length of sideLength
def getRandomSpins(sideLength, dimensions):
    #generate the appropriate array with a list comprehension
    spins=np.empty( [sideLength for i in range(dimensions)] )
    
    #iterate through the array, I don't really like how numpy does this
    for i in np.nditer(spins, op_flags=['writeonly']):
        #assign +1 or -1 to each element
        i[...] = 2*(np.random.random_integers(0,1) - 0.5)

    return spins

        

# ----------- Simulation/ Visualization Functions -------------- #
# Tkinter code to render the GUI and the necessary controls 

class IsingSquareVisual:
    def __init__(self, spins):
        self.findNeighbors = findNeighbors_2D
        self.isingModel = metropolis
        self.j       = 1
        self.temp    = 5
        self.spins   = spins
        self.closing = False
        
        # Set up the GUI with Tkinter
        self.root    = tk.Tk()
        self.controlFrame = tk.Frame(self.root)
        
        #temperature slider
        self.tScale  = tk.Scale(self.controlFrame, from_=0.1, to=10, resolution=.1,  label="Temperature")
        self.tScale.set(self.temp)
        
        #Exchange Parameter slider
        self.jScale  = tk.Scale(self.controlFrame, from_=-10, to=10, resolution=.1, label="J")
        self.jScale.set(self.j)
        
        # Triangle lattice checkbox
        self.triCheck= tk.Checkbutton(self.controlFrame, text="Triangle Latttice", command=self.toggleTriangle)
        
        # Model type button
        self.modelCheck = tk.Button(self.controlFrame, text="Switch to Wolff", command=self.toggleModel)
        
        # Reset button
        self.rButton = tk.Button(self.controlFrame, text="Reset", command=self.reset)
        
        # And lastly, the canvas we will draw the simulation to
        self.canvas  = tk.Canvas(self.root, bg="white", width=WINDOW_SIZE, height=WINDOW_SIZE, confine=True, bd=0)
        
        # Pack that all up
        self.canvas.pack()
        self.controlFrame.pack()
        self.jScale.pack(side="left")
        self.tScale.pack(side="left")
        self.rButton.pack(side="left")
        self.triCheck.pack(side="left")
        self.modelCheck.pack(side="left")
        
        # Probably necessary to ensure weird problems don't commence when you close the program
        self.root.protocol("WM_DELETE_WINDOW", self.closingHandler)
        
        # And start the animation
        self.root.after(0, self.animation)
        self.root.mainloop()
            
    def animation(self):
        self.temp = self.tScale.get()
        self.j    = self.jScale.get()
        if self.closing:
            # then the (x) button was pressed, so you should stop now
            self.root.quit()
            exit()
        else:
            self.drawFrame()
    
    def drawFrame(self):
        #clears canvas
        self.canvas.delete("all")
        
        #updates spins (this is the important part)
        self.isingModel(self.spins, self.temp, self.j, self.findNeighbors)

        #draws appropriate rectangles
        if(self.findNeighbors == findNeighbors_tri):
            self.drawTriangles()
        else:
            self.drawSquares()
                    
        self.canvas.update()
        #and call this function again in infinite recursion
        self.root.after(0, self.animation)
    
    # draws the representation of the atoms in a triangular lattice
    def drawTriangles(self):
        for x in range(SPIN_SIZE):
            for y in range(SPIN_SIZE):
                if(self.spins[x][y] > 0):
                    xC = 0
                    if(y%2 == 0):
                        xC = TRI_SIZE * x + TRI_SIZE / 2
                    else:
                        xC = TRI_SIZE * x + TRI_SIZE
                    yC = (np.sqrt(3)/2) * TRI_SIZE * (y + 1)
                    #coodinates of a hexagon with side length of HEX_SIZE centered at (xC, yC)
                    coords = [(xC               , yC - HEX_SIZE),
                              (xC + TRI_SIZE / 2, yC - HEX_SIZE / 2),
                              (xC + TRI_SIZE / 2, yC + HEX_SIZE / 2),
                              (xC               , yC + HEX_SIZE),
                              (xC - TRI_SIZE / 2, yC + HEX_SIZE / 2),
                              (xC - TRI_SIZE / 2, yC - HEX_SIZE / 2)]
                    self.canvas.create_polygon(coords, fill="black", width=0)
    
    # draws the representation of the atoms in a square lattice
    def drawSquares(self):
        for x in range(SPIN_SIZE):
            for y in range(SPIN_SIZE):
                if(self.spins[x][y]>0):
                    xC=x*BLOCK_SIZE
                    yC=y*BLOCK_SIZE
                    self.canvas.create_rectangle(xC, yC, xC+BLOCK_SIZE, yC+BLOCK_SIZE, fill="black", width=0)
    
    
    # GUI callback function for the triangle checkbox, 
    # toggles between a triangular lattice and a square lattice
    def toggleTriangle(self):
        self.reset()
        if(self.findNeighbors == findNeighbors_2D):
            self.findNeighbors=findNeighbors_tri
        else:
            self.findNeighbors=findNeighbors_2D
   
   # GUI callback called when the (x) button is pressed on the window
    def closingHandler(self):
        self.closing = True
        
    # GUI callback function for the model button, toggles between wolff and metropolis
    def toggleModel(self):
        self.reset()
        if(self.isingModel == metropolis):
            self.isingModel = wolff
            self.modelCheck.config(text="Switch to Metropolis")
        else:
            self.isingModel = metropolis
            self.modelCheck.config(text="Switch to Wolff")
    
    # GUI callback function for the reset button, and also just randomizes the spins
    def reset(self):
        for i in np.nditer(self.spins, op_flags=['writeonly']):
            #assign +1 or -1 to each element
            i[...] = 2*(np.random.random_integers(0,1) - 0.5)

# returns cluster size at a specific coordinate
def clusterAround(spins, findNeighbors, coord, recorded):
    recorded.append(coord)
    stack = findNeighbors(spins, coord)
    localSpin = spins.item(coord)
    clusterSize=1
    for i in stack:
        if(spins.item(i)==localSpin
           and not i in recorded):
            recorded.append(i)
            stack+=findNeighbors(spins, i)
            clusterSize+=1
    return clusterSize

# returns the average cluster size of an n-dimensional lattice defined by the findNeighbors function
def clusterSize(spins, findNeighbors):
    recorded=[]
    it=np.nditer(spins, flags=['multi_index'])
    totalClusters=0
    totalClusterSize=0
    while not it.finished:
        if(not it.multi_index in recorded):
            totalClusterSize+=clusterAround(spins, findNeighbors, it.multi_index, recorded)
            totalClusters+=1
        it.iternext()
    return totalClusterSize/float(totalClusters)
       
# And Finally! The code that runs.

if(len(sys.argv)  == 1):
    #If no arguments are used, just show visualization
    vis = IsingSquareVisual(getRandomSpins(SPIN_SIZE, 2))
else:
    findNeighbors = findNeighbors_2D
    model = metropolis
    #iterations is the number of iterations of the model per temperature
    iterations = 500
    # ouputFrequency is the number of iterations between an output of data
    outputFrequency = 100
    
    # parse command-line arguments, perhaps change some settings
    for arg in sys.argv:
        if(arg.startswith("-lattice=")):
            latticeStr=arg[9:]
            if(latticeStr == '1d'):
                DIMENSIONS = 1
                FREQUENCY = SPIN_SIZE ** 1
                findNeighbors=findNeighbors_1D
                
            elif(latticeStr == '3d'):
                DIMENSIONS = 3
                FREQUENCY = SPIN_SIZE ** 3
                findNeighbors=findNeighbors_3D
                
            elif(latticeStr == 'triangular' or latticeStr == 'tri'):
                findNeighbors=findNeighbors_tri
        
        elif(arg == '-model=wolff'):
            model=wolff
            
        elif(arg.startswith('-size=')):
            print 'set size to ',int(arg[6:])
            SPIN_SIZE = int(arg[6:])
            FREQUENCY = SPIN_SIZE ** 2
            
        elif(arg.startswith('-iterations=')):
            iterations = int(arg[12:])
            
        elif(arg.startswith('-outputFreq=')):
            outputFrequency = int(arg[12:])
    
    # Run simulation and save to file 
    # (we changed this part several times to modify the data produced)
    with open("magnetization.dat",'w') as f:
        temp = 5
        # in this iteration of our code, we generated data points around
        # the critical temperature by, if the final magnetization was
        # greater than 0.08, dividing temperature in two, and if it 
        # was less, multiplying it by 1.5 for the next iteration.
        # This, we found, made a good distrobution of data points that
        # gradually gave us better data, so we could look at it while it
        # was in the midst of calculating (which frequently took a long time)
        for i in range(100):
            f.write(str(temp)+' ')
            spins = getRandomSpins(SPIN_SIZE, DIMENSIONS)
            mag = 0
            sys.stderr.write(str(temp)+' ')
            # go to 500 iterations per temperature
            for iterations in range(iterations):
                model(spins, temp, 1, findNeighbors)
                if(iterations % outputFrequency == 0):
                    # record the magnetization every 100 iterations
                    mag=abs(magnetization(spins))
                    f.write(str(mag)+' ')
                    sys.stderr.write(str(mag)+' ')

            f.write('\n')
            sys.stderr.write('\n')
            
            # find next temperature to look at
            if(mag > 0.08):
                temp*=1.5
            else:
                temp*=.5  