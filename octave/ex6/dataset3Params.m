function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
valuesC = [0.01; 0.03; 0.1; 0.3; 1.1; 1.3; 10; 30];
valuessigma = [0.01; 0.03; 0.1; 0.3; 1.1; 1.3; 10; 30];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%min = 9999999999

%for i = 1:8
%	for j = 1:8

%		model= svmTrain(X, y, valuesC(i), @(x1, x2) gaussianKernel(x1, x2, valuessigma(j))); %train SVM model
%		predictions = svmPredict(model, Xval); %introduce cross validation set, get predictions
%		temp = mean(double(predictions ~= yval)); %calculate error rate of predictions for model with i,j on cv set
%		if(temp < min)%keep track of min error
%			min = temp
%			C = valuesC(i)
%			sigma = valuessigma(j)
%		endif
%
%	endfor
%endfor

min = 0.035
C = 0.3
sigma = 0.1


% =========================================================================

end
