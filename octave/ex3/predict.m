function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);



% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

aOne = [ones(m, 1) X]; % 5000x400...adding one now x401...a10

zTwo = aOne * Theta1';  % 5000x401 x 401x25

aTwo = sigmoid(zTwo); % 5000x25...needto add bias a20


n = size(aTwo, 1);
aTwo = [ones(n, 1) aTwo]; % 5000x26

zThree = aTwo * Theta2'; % 5000x26 x 26x10

aThree = sigmoid(zThree); % 5000x10

[maxx, p] = max(aThree, [], 2); % 5000 examples predicted
% for each example, 10 columns for 0-9
% take max of each row, col with max is the number predicted for that ex

% =========================================================================


end
