function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = X * theta; %12x2 x 2x1 gives 12x1

theta(1) = 0; %set theta0 (1 in octave) to 0 because we dont want to regularize this

regJ = (lambda / (2 * m)) * sum((theta .^ 2)); %get reg component for cost 1x1

J = (1 / (2 * m)) * sum((h - y) .^ 2) + regJ;%find regularized cost 1x1

regG = (lambda / m) * theta; %get reg component for grad, theta0 (1 in octave) is 0 so its not regularized

grad = (1/m) * X' * (h - y) + regG;%calc regularized grad 2x1




% =========================================================================

grad = grad(:);

end
