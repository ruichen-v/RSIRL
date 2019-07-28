function [cost, tau, D_tau_w, D_cost_w, D_tau_wc, D_cost_wc] = est_H(x, k, M, dynamics, disturbances,dist_last, counter, P_w, V_w, u_disc, w_c, beta, grad_compute, par, LP_sol, LP_prob)
% x: first frame of stage
% k: Look ahead index (0/1)
% P_w: A ('A') and b-w ('b') of envolope
% V_w: Vertices of envopole
% u_disc: dual-stage action space
% grad_compute: if gradient reaches final step (max look-ahead step)
% 

% return:
%           cost R      - expected cost of current stage
%           tau R^{n_A} - CRM of each action
%           D_tau_w     - (M x n_A) col j => D_tau[uj]_w
%           D_cost_w    - D_gtilda_w
%           D_tau_wc    - col j => D_tau[uj]_c
%           D_cost_wc   - D_gtilda_c

%% 
action_space = u_disc{k+1};

n_A = length(action_space);
tau = zeros(n_A, 1); % CRM of each action
D_tau_w = zeros(M, n_A); % Gradient of w with each possible current action
D_tau_wc = zeros(length(w_c),n_A); % Gradient of c with each possible current action

for j = 1:n_A
    u = action_space{j};
    [tau(j), D_tau_w(:,j), D_tau_wc(:,j)] = est_tau(x, u, k, M, dynamics, disturbances,dist_last, counter, P_w, V_w, u_disc, w_c, beta, grad_compute, par, LP_sol, LP_prob);
end 

%% apply softmax

tau_s = beta*tau;

sigma_H = exp(-tau_s)./sum(exp(-tau_s)); % prob of selecting each action
cost = sigma_H'*tau; 

D_cost_w = zeros(M,1);
D_cost_wc = zeros(length(w_c),1);

if (grad_compute)
    D_cost_w = D_tau_w*sigma_H;
    D_cost_wc = D_tau_wc*sigma_H;
end

end
