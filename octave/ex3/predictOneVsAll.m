function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       


[maxx, p] = max(sigmoid(X * all_theta'), [], 2);

% results in 5000 x 10

%5000 x 401   x 401 x 10

% ex1 sigmoid(theta0*x0 +...theta400*x400) = h1(ex1)
% ex2 sigmoid(theta0*x0 +...theta400*x400) = h1(ex2)
% ex5000 sigmoid(theta0*x0 +...theta400*x400) = h1(ex5000)

% ex1 sigmoid(theta0*x0 +...theta400*x400) = h2(ex1)
% ex2 sigmoid(theta0*x0 +...theta400*x400) = h2(ex2)
% ex5000 sigmoid(theta0*x0 +...theta400*x400) = h2(ex5000)

% ex1 sigmoid(theta0*x0 +...theta400*x400) = c
% ex2 sigmoid(theta0*x0 +...theta400*x400) = h10(ex2)
% ex5000 sigmoid(theta0*x0 +...theta400*x400) = h10(ex5000)

%	h1(ex1)	   h2(ex1)    h10(ex1)
%ex1	h1(ex1)    h2(ex1)    h10(ex1)
%ex2	h1(ex2)    h2(ex2)    h10(ex2)
%ex5000	h1(ex5000) h2(ex5000) h10(ex5000)

%this is 5000 x 10
% each row reps a training example, each col is probability that 
%training example is that digit
%using max of rows, we can get a prediction vector
%containing INDEX of MAX PROBABILITY for each ROW
%so highest hi(x) will determine what digit exm is...


% =========================================================================


end
