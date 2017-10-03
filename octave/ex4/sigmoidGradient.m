function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).


g = sigmoid(z) .* (1 - sigmoid(z));



%g is shape of z...use elementwise

%S1 = 400, S2 = 25
%X with bias is 5000x401
%theta must be 401x25
%z2 = theta1 * a1 where a1=x1
%z2 = a1 * theta1' 5000x401 x 401x25 so z2 is 5000x25
%a2 = g(z2) ...a2 is still 5000x25...ad bias to a2 makes it 26
%z3 = a2 * theta2' 5000x26 x 26x10	theta is 26x10, 10 outputs
%a3 = g(z3) is still 5000x10	
%so g doesnt change shape here...



% =============================================================




end
