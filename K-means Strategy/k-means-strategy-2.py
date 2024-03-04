from Precode2 import *
import numpy
import matplotlib.pyplot as plt

data = np.load('AllSamples.npy')
k1,i_point1,k2,i_point2 = initial_S2('6128') # please replace 0111 with your last four digit of your ID
print(k1)
print(i_point1)
centroids = initialize_centroids(data, i_point1, k1)
data_centroids = np.full_like(data, 0.0)
prev_i_point1 = (centroids + 0.5).copy()
loss = np.empty([9,1])
print("Initialized centroids: ", centroids, "\n")
i = 0
while (prev_i_point1 != centroids).any():
    prev_i_point1 = centroids.copy()
    data_centroids = assign_centroids(data, centroids, data_centroids)
    loss[i] = objective(data, centroids, data_centroids)
    print("error ", loss[i])
    centroids = compute_centroids(data, centroids, data_centroids)
    print("Centroids are: ", centroids, "\n")
    i += 1
print("Final centroids are: ", centroids, "\n")
print("Loss is: ", objective(data, centroids, data_centroids))
x = np.arange(0, 9)
plt.title("Strategy 2 Loss for k1 = 4")
plt.xlabel("K")
plt.ylabel("Loss")
plt.plot(x, loss, color="green")

plt.show()
print(k2)
print(i_point2)
centroids_2 = initialize_centroids(data, i_point2, k2)
data_centroids = np.full_like(data, 0.0)
prev_i_point2 = (centroids_2 + 0.5).copy()
loss = np.empty([13,1])
print("Initialized centroids: ", centroids_2, "\n")
i = 0
while (prev_i_point2 != centroids_2).any():
    prev_i_point2 = centroids_2.copy()
    data_centroids = assign_centroids(data, centroids_2, data_centroids)
    loss[i] = objective(data, centroids_2, data_centroids)
    centroids_2 = compute_centroids(data, centroids_2, data_centroids)
    print("Centroids are: ", centroids_2, "\n")
    i += 1
print("Final centroids are: ", centroids_2, "\n")
print("Sum of squared errors is: ", objective(data, centroids_2, data_centroids))
x = np.arange(0, 13)
plt.title("Strategy 2 Loss for k2 = 6")
plt.xlabel("K")
plt.ylabel("Loss")
plt.plot(x, loss, color="purple")

plt.show()

#Computes the objective function of the data.
#I made use of a separate array called “data_centroids” which has the same shape as data and where data_centroids[i] is
#the centroid that data[i] belongs in. I then iterated through all the points in data, and for each point, I iterated
#through all the centroids until the centroid is the one point was assigned to
#((data_centroids[i] == centroid).all()). Finally, I computed the squared error of the point and centroid and added it
#to the sum variable.

def objective(data, centroids, data_centroids):
    sum = 0
    i = 0
    for point in data:
        for centroid in centroids:
            if (data_centroids[i] == centroid).all():
                error = centroid - point
                squared_error = error[0]**2 + error[1]**2
                sum += squared_error
        i += 1
    return sum

#Computes the euclidean distance between a point and a centroid.

def euclidean_distance(point, centroid):
    euclidean_distance = ((point[0] -centroid[0])**2 + (point[1] - centroid[1])**2)**0.5
    return euclidean_distance

#First, I iterate from 1 to k to pick k centroids. For every point, I calculate the average distance to the other
#i-1 centroids. Whichever point has the maximum average distance will be initialized as a centroid. It’s also
#important to note that I must skip a point with continue if the point chosen is already a centroid.

def initialize_centroids(data, initial_centroid, k):
    centroids = numpy.empty([k, 2])
    centroids[0] = initial_centroid
    for i in range(1, k):
        max_average_distance = 0
        for point in data:
            if point in centroids:
                continue
            total_distance = 0
            for centroid in centroids:
                total_distance += euclidean_distance(point, centroid)
            average_distance = total_distance/i
            if (max_average_distance < average_distance):
                max_average_distance = average_distance
                candidate_centroid = point
        centroids[i] = candidate_centroid
    return centroids

#For each centroid, I check which data points belong to the centroid and add it to a variable named total_point.
#I divide total_point by the number of points and returned a vector temp to the array that had the
#initial points called i_point1.

def compute_centroids(data, centroids, data_centroids):
    temp = centroids
    for c, centroid in enumerate(centroids):
        i = 0
        total_point = np.full_like(centroid, 0)
        count = 0
        for point in data:
            if (data_centroids[i] == centroid).all():
                total_point += point
                count += 1
            i += 1
        temp_centroid = total_point/count
        temp[c] = temp_centroid
    return temp
