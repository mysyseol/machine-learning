clear, clc
n = 100; D = 2; K = 3; 
X = zeros(n*K,D);
y = zeros(n*K,1);
for j=1:K    
    ix = n*(j-1)+1 : n*(j);
    r = linspace(0., 1., n)';
    t = linspace((j-1)*4, j*4, n)' + randn(n,1)*0.2;
    X(ix,:) = [r.*sin(t), r.*cos(t)];
    y(ix) = j-1;
end
plot(X(:,1), X(:,2),'.')

% train a Linear Classifier

% initialize parameters randomly
h = 100;
W = 0.01*randn(D,h);
b = zeros(1,h);
W2 = 0.01*randn(h,K);
b2 = zeros(1,K);

% some hyperparameters
step_size = 1e-0;
reg = 1e-3; % regularization strength

% gradient descent loop
num_examples = size(X,1);
fprintf('start training\n')
for i_step = 1:10000
    
    % evaluate class scores, [N*K]    
    hidden_layer = X*W + repmat(b, num_examples,1);
    hidden_layer(hidden_layer<=0)=0;
    scores = hidden_layer*W2 + repmat(b2, num_examples,1);
    
    % compute the class probabilities
    exp_scores = exp(scores);
    probs = exp_scores ./ repmat(sum(exp_scores,2),1,K); % [N*K]
    
    corect_logprobs = zeros(num_examples,1);
    for i=1:num_examples
        j = y(i) + 1;
        corect_logprobs(i) = -log(probs(i,j));
    end
    
    % compute the loss : average cross-entropy loss and regularization
    data_loss = sum(corect_logprobs)/num_examples;
    reg_loss = 0.5*reg*sum(sum(W.*W));
    loss = data_loss + reg_loss;
    
    if mod(i_step, 1000) == 0
        fprintf('iteration %d \t loss = %f\n', i_step, loss)
    end
    
    % compute the gradient on the scores
    d_scores = probs;
    for i=1:num_examples
        j = y(i) + 1;
        d_scores(i,j) = d_scores(i,j) - 1;
    end
    d_scores = d_scores/num_examples;
    
    % back-propagate the gradient to the parameters
    % first back-prop into parameters W2 and b2
    d_W2 = transpose(hidden_layer) * d_scores;
    d_b2 = sum(d_scores);
    % next backprop into hidden layer
    d_hidden = d_scores * transpose(W2);
    % backprop the ReLU non-linearity
    d_hidden(hidden_layer <= 0) = 0;
    % finally into W, b
    d_W = transpose(X) * d_hidden;
    d_b = sum(d_hidden);
    % add regularization gradient contribution
    d_W2 = d_W2 + reg*W2;    
    d_W  = d_W + reg*W;
    
    % perform a parameter update
    W  = W  - step_size * d_W;
    b  = b  - step_size * d_b;
    W2 = W2 - step_size * d_W2;
    b2 = b2 - step_size * d_b2;
end
fprintf('end of training\n')
% evaluate training set accuracy
hidden_layer = X*W + repmat(b, num_examples,1);
hidden_layer(hidden_layer<=0)=0;
scores = hidden_layer*W2 + repmat(b2, num_examples,1);

[p_, predicted_class] = max(scores,[],2);
accuracy = mean(predicted_class == y+1);
fprintf('training accuracy : %f\n', accuracy)
