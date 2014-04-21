function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

sumatory1=0;

for i=1:m
	sumatory1+= (-y(i)*log(sigmoid((theta'*(X(i,:)'))))-(1-y(i))*log(1-sigmoid((theta'*(X(i,:)')))));
end

sumatory2=0;

for i=2:size(theta,1)
	sumatory2+= theta(i)^2;
end

J = inv(m)*sumatory1 + lambda/(2*m)*sumatory2;

for i=1:size(theta,1)

	sumatory=0;
	
	if (i==1)
		for j=1:m
			sumatory+=(sigmoid(theta'*(X(j,:)'))-y(j))*X(j,i);
		end
		grad(i)=inv(m)*sumatory;
	else
		for j=1:m
			sumatory+=(sigmoid(theta'*(X(j,:)'))-y(j))*X(j,i);
		end
		grad(i)=inv(m)*sumatory+lambda*theta(i)/m;
	endif
end


% =============================================================

end
