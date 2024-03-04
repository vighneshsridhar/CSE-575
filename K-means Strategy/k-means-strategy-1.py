from Precode import *
import numpy
import random
import matplotlib.pyplot as plt


data = np.load('AllSamples.npy')
k1,i_point1,k2,i_point2 = initial_S1('6128') # please replace 0111 with your last four digit of your ID
print(k1)
print(i_point1)
data_centroids = np.full_like(data, 0.0)
prev_i_point1 = (i_point1 + 0.5).copy()
print(i_point1)
print(prev_i_point1, "\n")
loss = np.empty([6,1])
i = 0
while (prev_i_point1 != i_point1).any():
    prev_i_point1 = i_point1.copy()
    data_centroids = assign_centroids(data, i_point1, data_centroids)
    loss[i] = objective(data, i_point1, data_centroids)
    i_point1 = compute_centroids(data, i_point1, data_centroids)
    print("Centroids are: ", i_point1, "\n")
    i += 1
print("Final centroids are: ", i_point1, "\n")
print("Sum of squared errors is: ", objective(data, i_point1, data_centroids))


x = np.arange(0, 6)
plt.title("Strategy 1 Loss for k1 = 3")
plt.xlabel("K")
plt.ylabel("Loss")
plt.plot(x, loss, color="red")

plt.show()


print(k2)
print(i_point2, "\n")
data_centroids = np.full_like(data, 0.0)
prev_i_point2 = (i_point2 + 0.5).copy()
loss = np.empty([9,1])
i = 0
while (prev_i_point2 != i_point2).any():
    prev_i_point2 = i_point2.copy()
    data_centroids = assign_centroids(data, i_point2, data_centroids)
    loss[i] = objective(data, i_point2, data_centroids)
    print ("error ",loss[i])
    i_point2 = compute_centroids(data, i_point2, data_centroids)
    print("Centroids are: ", i_point2, "\n")
    i += 1
print("Final centroids are: ", i_point2, "\n")
print("Sum of squared errors is: ", objective(data, i_point2, data_centroids))
x = np.arange(0, 9)
plt.title("Strategy 1 Loss for k2 = 5")
plt.xlabel("K")
plt.ylabel("Loss")
plt.plot(x, loss, color="blue")

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

#In assign_centroids, for each point, I calculated the Euclidean distance from the point to each centroid.
#The centroid which was closest to the point was assigned. In my code, this is written as
#data_centroids[i] = min_centroid, where i represents the i-th data point.

def assign_centroids(data, centroids, data_centroids):
    i = 0
    for point in data:
        min_distance = 100.0
        for centroid in centroids:
            distance = euclidean_distance(point, centroid)
            if distance < min_distance:
                min_distance = distance
                min_centroid = centroid
        data_centroids[i] = min_centroid
        i += 1
    return data_centroids

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
