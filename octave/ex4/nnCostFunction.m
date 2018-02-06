function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

aOne = [ones(size(X, 1), 1) X]; %add bias X10	5000x401

zTwo = aOne * Theta1'; %5000x401 x 401x25

aTwo = sigmoid(zTwo); % 5000x25

aTwo = [ones(size(aTwo, 1), 1) aTwo]; %add biasX20	5000x26

zThree = aTwo * Theta2'; % 5000x26 x 26x10

aThree = sigmoid(zThree); % 5000x10

h = aThree;

%for each training example, sum up the columns...

%yv = [1:num_labels] == y;
%eye(num_labels)([1;2;4],:) just a way to index...will give [1 0 0 0 0 0 0 0 0 0 0; 0 1 0 0 0 0 0  0 0 0 0 ; 0 0 0 1 0 0 0 0 0 0 0 ]...10 col/row

yv= eye(num_labels)(y,:); % set y as vector representing output digit y = 5 = [0 0 0 0 1...]


J = (1 / m) * sum(sum([-yv .* log(h) - (1 - yv) .* log(1 - h)]));

regTerm = (lambda / (2 * m)) * [sum(sum(Theta1(:, 2:size(Theta1,2)) .* Theta1(:, 2:size(Theta1,2)))) + sum(sum(Theta2(:, 2:size(Theta2,2)) .* Theta2(:, 2:size(Theta2,2))))];

J = J + regTerm;




%part 1
%MUST USE elementwise!!

%yik is a number, hxik is a number
%so you have two matrices 5000 x 10
%

% yi1.....yi10       .* log(h(xi1)1)........log(h(xi)10) or 1-hx depends on y

%
%multiplying one y with one ^^ gives a val at i and k
%then you want to sum up all of the k ys for that ex
%then after doing it for all of the classes for the examples, sum up all 5000 

%part 1 regularization
%use elementwise again, exclude theta 0 param
%then add regterm to cost


deltaThree = h - yv; % 5000x10 - 5000x10 for each training example, for each output node for each training example, measure difference between networks activation and true target value

%theta2' * deltaThree .* g'(zTwo).....10x26	5000x10    g'(z2) is 5000x25 remove bias deltaTwozero
deltaTwo = deltaThree * Theta2(:,2:size(Theta2,2)) .* sigmoidGradient(zTwo); %deltaTwo is 5000x25


grad1 = deltaTwo' * aOne; % 25x5000 x 5000x401
grad2 = deltaThree' * aTwo; %10x5000 x 5000x26

grad1(:,2:size(grad1,2)) = grad1(:,2:size(grad1,2)) + (lambda) * Theta1(:,2:size(Theta1,2));
grad2(:,2:size(grad2,2)) = grad2(:,2:size(grad2,2)) + (lambda) * Theta2(:,2:size(Theta2,2));

Theta1_grad = grad1 / m;
Theta2_grad = grad2 / m;









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
