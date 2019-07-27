function Prob = setup_LP_yalmip(L,A_P,b_U_p)

yalmip('clear');

%% variables
M = length(b_U_p); % 8

q = sdpvar(M*L,1);
b_global = sdpvar(M*M,1);
costs_global = sdpvar(M*L,1);

%% Constraints

A_global = kron(eye(M),A_P);

Constraints = [];
for j = 1:M
    Constraints = [Constraints;sum(q(1+(j-1)*L:j*L))==1]; % q is distribution
end

Constraints = [Constraints;
               q >= 0;
               A_global*q <= b_global]; % semi-param envolopes

%% Define problem

Objective = costs_global'*q; % sum of multiple CRMs with different offset b

Prob = optimizer(Constraints,Objective,sdpsettings('solver','mosek'),{b_global,costs_global},q); % vary {}, get optimal q

end

