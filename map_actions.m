function [actions, opt_action] = map_actions(u_expert, means)
% maps to an action from continuous space to the closest mean action, using
% Euclidean distance

nb_actions = length(u_expert); % all executed operations
actions = {};
opt_action = zeros(nb_actions, 1);

K = size(u_expert{1},2); % Operation length
nb_means = size(means, 1); % Number of discrete operations
for i=1:nb_actions
    u_i = u_expert{i};
    u_i = [u_i(1,:), u_i(2,:)]; % flatten to compare
    dist = zeros(nb_means, 1); % dist to each cluster mean
    for j=1:nb_means
        dist(j,1) = norm(u_i - means(j,:));
    end
    [~, idx] = min(dist);
    opt_action(i,1) = idx; % classification indices
    actions{i} = [means(idx,1:K); means(idx,K+1:2*K)]; % nearest op for each user op
end

end


