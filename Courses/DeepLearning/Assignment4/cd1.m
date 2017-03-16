function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
    %error('not yet implemented');
    % data to hidden prob
    visible_data = sample_bernoulli(visible_data);
    
    hidden_probability_1 = logistic(rbm_w * visible_data);
    hidden_state_1 = sample_bernoulli(hidden_probability_1);
    
    grad_1 = configuration_goodness_gradient(visible_data, hidden_state_1);
    
    % reconstruction
    visible_probability_2 = logistic(rbm_w' * hidden_state_1);
    visible_state_1 =  sample_bernoulli(visible_probability_2);
    
    % reconstruction to hidden
    hidden_probability_2 = logistic(rbm_w * visible_state_1);
    %hidden_state_2 = sample_bernoulli(hidden_probability_2);
    hidden_state_2 = hidden_probability_2;
    
    grad_2 = configuration_goodness_gradient(visible_state_1, hidden_state_2);
    
    ret = grad_1 - grad_2;
    
    
end
