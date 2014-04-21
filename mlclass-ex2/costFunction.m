function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%


sumatory=0;

for i=1:m
	sumatory+= (-y(i)*log(sigmoid((theta'*(X(i,:)'))))-(1-y(i))*log(1-sigmoid((theta'*(X(i,:)')))));
end

J = inv(m)*sumatory;

sumatory=0;

for i=1:size(theta,1)

	sumatory=0;

	for j=1:m
		sumatory+=(sigmoid(theta'*(X(j,:)'))-y(j))*X(j,i);
	end
	
	grad(i)=inv(m)*sumatory;	
end


% =============================================================

end
