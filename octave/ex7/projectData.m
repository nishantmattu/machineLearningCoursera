function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

for i = 1:size(Z,1)
	for k = 1:K
		x = X(i, :)';%mxn' for 1xn...x becomes nx1...reps a training example
		Z(i,k) = x' * U(:, k); %1xn * nxk...with Ureduced and X member of n, Z becomes 1xk

	%hm...for every k...can vectorize?
	%
	endfor
endfor

%z is mxk


% =============================================================

end
