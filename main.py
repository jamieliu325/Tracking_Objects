import numpy as np
import cv2

# Repeatability
np.random.seed(0)

# GLOBAL VARIABLE
VFILENAME = "walking.mp4"
# set boundary within where particles will be generated
HEIGHT = 406
WIDTH = 722
# number of particles will be generated
NUM_PARTICLES = 150
# moving velocity of particles
VEL_RANGE = 1
# colour array for the tracking object
TARGET_COLOUR = np.array((156,74,38))
# standard deviation for position and velocity
POS_SIGMA = 1
VEL_SIGMA = 0.5

# load video frames from file
def get_frames(filename):
    # open video file
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        # capture frame by frame, if frame is read correctly ret is True
        ret, frame = video.read()
        if ret:
            yield frame
        else:
            break
    # close video file
    video.release()
    yield None

#  creating a particle cloud
def initialize_particles():
    # create an array in NUM_PARTICLES * 4
    particles = np.random.rand(NUM_PARTICLES,4)
    # first col: x position (WIDTH), second col: y position (HEIGHT), last two cols: velocity for x and y
    particles = particles * np.array((WIDTH,HEIGHT,VEL_RANGE,VEL_RANGE))
    # particles can be either positive or negative
    particles[:,2:4] -= VEL_RANGE/2
    return particles

#  moving particles according to their velocity state
def apply_velocity(particles):
    particles[:,0] += particles[:,2]
    particles[:,1] += particles[:,3]
    return particles

# Prevent particles from falling off the edge of the video frame
def enforce_edges(particles):
    for i in range(NUM_PARTICLES):
        # particle will fall within 0 to WIDTH-1 for x position and 0 to HEIGHT-1 for y position
        particles[i,0] = max(0, min(WIDTH-1, particles[i,0]))
        particles[i,1] = max(0, min(HEIGHT-1, particles[i,1]))
    return particles

# Measure each particle's quality
def compute_errors(particles,frame):
    # create initial array with 0s
    errors = np.zeros(NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        x = int(particles[i,0])
        y = int(particles[i,1])
        # get the value of pixel of frame at (x, y)
        pixel_colour = frame[y,x,:]
        # save the errors between pixel values of each particles to the target colour in array
        errors[i] = np.sum((TARGET_COLOUR - pixel_colour)**2)
    return errors

# assign weights to the particles based on their quality of match
def compute_weights(errors):
    # the larger the weight is, the closer the particles are to the target object
    weights = np.max(errors) - errors
    # give 0 weight to the points at edge
    weights[
        (particles[:,0] == 0) | (particles[:,0] == WIDTH - 1) | (particles[:,1] == 0) | (particles[:,1] == HEIGHT - 1)
    ] = 0.0
    # get large weights larger
    weights **= 4
    return weights

# resample particles according to their weights
def resample(particles,weights):
    # normalize the weights and use it as probabilities
    probabilities = weights/np.sum(weights)
    # resample the particles based on probabilities
    index_numbers = np.random.choice(
        NUM_PARTICLES,
        size=NUM_PARTICLES,
        p=probabilities
    )
    # get a new set of particles after resampling
    particles = particles[index_numbers,:]
    # give the best guess by calculating the mean for x and y position
    x = np.mean(particles[:,0])
    y = np.mean(particles[:,1])
    return particles, (int(x), int(y))

# fuzz hte particles
def apply_noise(particles):
    # join the arrays to have the same shape as particles
    noise = np.concatenate(
        (
            # random samples from a normal distribution for x,y's position and velocity
            np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES, 1)),
            np.random.normal(0.0, POS_SIGMA, (NUM_PARTICLES, 1)),
            np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES, 1)),
            np.random.normal(0.0, VEL_SIGMA, (NUM_PARTICLES, 1))
        ),
        axis=1
    )
    # add noise to particles
    particles += noise
    return particles

# Display the video frames
def display(frame,particles,location):
    if len(particles) > 0:
        for i in range(NUM_PARTICLES):
            x = int(particles[i,0])
            y = int(particles[i,1])
            # create green circles at (x,y) on frame to visualize particles
            cv2.circle(frame, (x,y), 1, (0,255,0), 1)
    if len(location) > 0:
        # create one red circle at possible location of target object on frame to track the object
        cv2.circle(frame, location, 15, (0,0,255), 5)
    # display an image with circles in a window
    cv2.imshow('tracking', frame)
    # to display a window for given milliseconds
    cv2.waitKey(30)


# Main routine
particles = initialize_particles()
for frame in get_frames(VFILENAME):
    if frame is None:
        break
    particles = apply_velocity(particles)
    particles = enforce_edges(particles)
    errors = compute_errors(particles, frame)
    weights = compute_weights(errors)
    particles, location = resample(particles, weights)
    particles = apply_noise(particles)
    display(frame, particles, location)
# destroy all windows being created
cv2.destroyAllWindows()