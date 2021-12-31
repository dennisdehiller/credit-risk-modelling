%% 1.1
clc
[s,b] = bs(135, 95, 0, 2, 0.2, 0.02)

%% 1.2
clc
list95 = (0.05:0.01:0.3);
list70 = (0.05:0.01:0.3);
for i = 1:length(list95)
    list95(i) = spread(110, 95, 0, 2, list95(i), 0.03);
    list70(i) = spread(110, 70, 0, 2, list70(i), 0.03);
end

vol = (0.05:0.01:0.3);

plot(vol, list95, '--');
hold on
ylim([0, 0.07])
title('Yield spread as a function of volatility: V=110, T=2, r=0.02', 'FontSize', 10)
xlabel('\sigma', 'FontSize', 10) 
ylabel('Yield Spread', 'FontSize', 10) 
plot(vol, list70);
legend({'D = 95','D = 70'},'Location','northeast', 'FontSize', 12)
print -deps yield_spread.eps

%% 1.3
clc
senior = 150 - bs(150, 75, 0, 3, 0.35, 0.02)
equity = bs(150, 135, 0, 3, 0.35, 0.02)
junior = bs(150, 75, 0, 3, 0.35, 0.02) - equity
total = senior + equity + junior

%% 1.4
clc
mu = (0:0.02:0.5);
for i = 1:length(mu)
    mu(i) = defprob(200, 150, mu(i), 0.3, 5);
end

xval = (0:0.02:0.5);
plot(xval, mu);
title('The default probability in the Merton model: D=150, V=200, \sigma=0.3', 'FontSize', 12)
xlabel('\mu', 'FontSize', 12) 
ylabel('Default probability', 'FontSize', 12) 
print -deps default_prob_drift.eps

pd = defprob(200, 150, 0.3, 0.3, 5)

%% 1.5
clc

df_prob_1yr = 0.0000592193082343817;
df_prob_2yr = 0.00132865468919552;
df_prob_3yr = 0.00395401984041377;
df_prob_4yr = 0.00726621165935282;
df_prob_5yr = 0.0113661280257741;
probs_ev = [df_prob_1yr, df_prob_2yr, df_prob_3yr, df_prob_4yr, df_prob_5yr];
probs_cap = [df_prob_1yr, df_prob_2yr, df_prob_3yr, df_prob_4yr, df_prob_5yr];
total_debt = 80.065;
vol = 0.294;
mkt_cap = 275.6;
ev = 326.67;
quota_ev = total_debt / ev;
quota_cap = total_debt / (mkt_cap + total_debt)

for i = 1:5
    probs_ev(i) = ((log(quota_ev) - norminv(probs_ev(i)) * (vol * sqrt(i))) / i) + (0.5*vol^2);
end
probs_ev

plot([1,2,3,4,5], probs_ev)
hold on

xticks([1,2,3,4,5])
title('Implied \mu using Bloomberg DRSK default probabilities on H&M', 'FontSize',  12)
xlabel('T', 'FontSize', 12) 
ylabel('Implied \mu', 'FontSize', 12)

print -deps yield_spread.eps

for i = 1:5
    probs_cap(i) = ((log(quota_cap) - norminv(probs_cap(i)) * (vol * sqrt(i))) / i) + (0.5*vol^2);
end
probs_cap
plot([1,2,3,4,5], probs_cap, '--')

legend({'V = Market Cap + Debt','V = Enterprise Value'},'Location','southeast', 'FontSize', 10)


%% 1.6
clc
clear

mkt_cap = 283017.31;
vol = 0.2951;
lt_d = 54672;
st_d = 12844;
adj_cfo = 51502.06;
int_exp = 1299.06;
v0 = mkt_cap + lt_d + st_d;
T = 4;

quota_cap = (lt_d + st_d)/v0

%Estimate mu using the two closest periods
mu_values = [-0.0030, 0.0445];
mu = mean(mu_values);

def_cap = normcdf((log(quota_cap) - (mu - 0.5*(vol^2))*T)/ (vol*sqrt(T)))

lt_d = 54672;
st_d = 12844;
vix_ma_5yr = 17.06;
vix_ma_1yr = 22.23;
vix_multiple = vix_ma_1yr / vix_ma_5yr;
quota_cap = (lt_d + st_d)/v0

def_cap = normcdf((log(quota_cap) - (mu - 0.5*((vol*vix_multiple)^2))*T)/ ((vol*vix_multiple)*sqrt(T)))


%% 2.1
clc
t = (1:10);
table = zeros(4, 10);
price_rf = [0.9784  0.9336  0.8776  0.8295  0.7807  0.7323  0.6852  0.6398  0.5965  0.5554];
price_bond = [0.9758  0.9253  0.8600  0.7884  0.7162  0.6466  0.5815  0.5216  0.4672  0.4179];
recovery_rates = [0.05 0.25 0.4 0.7];

for i = 1:length(recovery_rates)
    for j = 1:length(price_bond)
        table(i, j) = ((price_bond(j) / price_rf(j)) - recovery_rates(i)) / (1 - recovery_rates(i));
    end
end

phi5 = table(1,:);
phi40 = table(3,:);

for i = 1:(length(phi5) - 1)
    phi5(i) = table(1, (i)) - table(1, i+1);
    phi40(i) = table(3, (i)) - table(3, i+1);
end

phi5(1:9);
phi40(1:9);

test_array = (1:10);

for i = 1:10
    test_array(i) = price_bond(i) / price_rf(i);
end

%% 2.2
clc
clear

lambdas = (0.025:0.025:0.5);
x_vals = (0.025:0.025:0.5);

for i = 1:20
   lambdas(i) = DefaultableCouponBond([lambdas(i) lambdas(i) lambdas(i) lambdas(i) lambdas(i) lambdas(i)], [1 2 3 4 5 6], [1 2 3 4 5 6], [1 1 1 1 1 1], 0.05);
end

plot(x_vals, lambdas)
title('Price of defaultable coupon-bearing bond with different intensities \lambda', 'FontSize', 10)
xlabel('\lambda', 'FontSize', 12) 
ylabel('Price', 'FontSize', 12) 

%%
clc
clear

interest = (0.01:0.01:0.2);
x_vals = (0.01:0.01:0.2);

for i = 1:20
   interest(i) = DefaultableCouponBond([0.05 0.05 0.05 0.05 0.05 0.05], [1 2 3 4 5 6], [1 2 3 4 5 6], [1 1 1 1 1 1], interest(i));
end

plot(x_vals, interest)
title('Price of defaultable coupon-bearing bond with different interest rates', 'FontSize', 10)
xlabel('Interest rate', 'FontSize', 12) 
ylabel('Price', 'FontSize', 12) 


%% 2.3

%% 2.4
clc
clear
% Vasicek

values_mu = (0.0:0.01:0.4);
values_x = (0.0:0.01:0.4);
for i = 1:41
    values_mu(i) = Vasicek([1,1,1], [1,2,3], values_mu(i), 0.3, 0.02, 0.5, 1, 0.05);
end

plot(values_x, values_mu)
title('Price of defaultable coupon-bearing bond with different \mu', 'FontSize', 12)
xlabel('\mu', 'FontSize', 12) 
ylabel('Price', 'FontSize', 12) 

%% alpha-graph
clc
clear 
values_a = (0.05:0.05:1);
values_x = (0.05:0.05:1);
for i = 1:20
    values_a(i) = Vasicek([1,1,1], [1,2,3], 0.05, 0.3, 0.02, values_a(i), 1, 0.05);
end

plot(values_x, values_a)
title('Price of defaultable coupon-bearing bond with different \alpha', 'FontSize', 12)
xlabel('\alpha', 'FontSize', 12) 
ylabel('Price', 'FontSize', 12) 

%% 3.1
clc
p = [0.5 0.35 0.10 0.05; 0.35 0.3 0.25 0.10; 0.35 0.25 0.25 0.15; 0 0 0 1];
a = [0.4 0.5 0.1 0];

% a
p2 = a * (p^2)
p3a = a * (p^3)
p7 = a * (p^7)

% b
invp1 = 1-(a * p)
invp4 = 1-(a * p^4)
invp6 = 1-(a * p^6)

% c
p3 = a * (p^3);
p3c = p3(3) + p3(4)

% d
a = [0 1 0 0];

p3d = a * (p^3)

%% 3.2
clear
clc
Q = [-0.1055 0.0704 0.0351; 0.242 -0.329 0.087; 0 0 0];
a = [0.5 0.5 0];

% a
p2a = a * (expm(Q*2))
p3a = a * (expm(Q*3))
p7a = a * (expm(Q*7))

% b
p1inv = 1 - (a * (expm(Q)))
p4inv = 1 - (a * (expm(Q * 4)))
p6inv = 1 - (a * (expm(Q * 6)))

% c
p35 = a * (expm(Q*3.5));
p35 = p35(2) + p35(3)

% d
a = [0 1 0];
p3d = a * (expm(Q*3))

%% 3.3
clc
clear
Q = [-0.1055, 0.0704, 0.0351; 0.242, -0.329, 0.087; 0, 0, 0];
alpha_aaa = [1, 0, 0];
alpha_aa = [0, 1, 0];

T = [1, 3, 5, 7, 10];
rf = 0.03;

prob_aaa = zeros(5, 3);
prob_aa = zeros(5, 3);

for i = 1:5
    prob_aaa(i,:) = alpha_aaa * (expm(T(i) * Q));
end

for i = 1:5
    prob_aa(i,:) = alpha_aa * (expm(T(i) * Q));
end

payout_aaa_10 = prob_aaa;
payout_aaa_50 = prob_aaa;
payout_aa_10 = prob_aa;
payout_aa_50 = prob_aa;

price_aaa_10 = zeros(1,5);
price_aaa_50 = zeros(1,5);
price_aa_10 = zeros(1,5);
price_aa_50 = zeros(1,5);

for i = 1:5
    payout_aaa_10(i, 3) = payout_aaa_10(i, 3) * 0.1;
    payout_aaa_50(i, 3) = payout_aaa_50(i, 3) * 0.5;
    payout_aa_10(i, 3) = payout_aa_10(i, 3) * 0.1;
    payout_aa_50(i, 3) = payout_aa_50(i, 3) * 0.5;

    price_aaa_10(i) = sum(payout_aaa_10(i,:));
    price_aaa_50(i) = sum(payout_aaa_50(i,:));
    price_aa_10(i) = sum(payout_aa_10(i,:));
    price_aa_50(i) = sum(payout_aa_50(i,:));
end

price_aaa_10
price_aaa_50
price_aa_10
price_aa_50


%% 4.1
clc
clear

m = 1000;
loansize = 1000000;
losses = 0.6;
p_bar = [0.04, 0.05, 0.1, 0.15];
rho = [0.1, 0.15, 0.35, 0.6];
x40 = 40000000;
x90 = 90000000;
loan_loss = losses*loansize;

loss_dist_40 = normcdf((1/sqrt(rho(1,1)))*(sqrt(1-rho)*norminv(x40/(loan_loss*m))-norminv(p_bar(1,1))));
loss_dist_90 = normcdf((1/sqrt(rho(1,1)))*(sqrt(1-rho)*norminv(x90/(loan_loss*m))-norminv(p_bar(1,1))));

% 40 < x < 90
x4090 = loss_dist_90 - loss_dist_40;
x4090(1,1)

for i = 2:4
    for j = 2:4
        loss_dist_40(i, j) = normcdf((1/sqrt(rho(i)))*(sqrt(1-rho(i))*norminv(x40/(loan_loss*m))-norminv(p_bar(j))));
        loss_dist_90(i, j) = normcdf((1/sqrt(rho(i)))*(sqrt(1-rho(i))*norminv(x90/(loan_loss*m))-norminv(p_bar(j))));
        x4090 = loss_dist_90 - loss_dist_40;
    end
end

x4090(2:4, 2:4)

%% 4.2
clc
clear

m = 1000;
loansize = 1;
losses = 0.6;
loan_loss = loansize*losses;
p_bar_list = [0.05, 0.1, 0.15];
rho_list = [0.15, 0.35, 0.6];
alpha_list = [0.95, 0.99, 0.999];
result_table_var = zeros(3,3,3);
result_table_es = zeros(3,3,3);

for i = 1:3
    alpha = alpha_list(i);    
    for j = 1:3
        p_bar = p_bar_list(j);
        for k = 1:3
            rho = rho_list(k);

            F_inv = (normcdf(((sqrt(rho)*norminv(alpha))+norminv(p_bar))/(sqrt(1-rho))));
            VaR = loan_loss*m*F_inv;

            inv = @(alpha) (normcdf(((sqrt(rho)*norminv(alpha))+norminv(p_bar))/(sqrt(1-rho))));

            ES = ((loan_loss*m)/(1-alpha))*integral(inv, alpha, 1);

            result_table_var(i, j, k) = VaR/1000;
            result_table_es(i, j, k) = ES/1000;


        end
    end
end
result_table_var;
result_table_es;

%% 4.4
clc
clear
rho_list = [0.15, 0.40, 0.80];
p_bar_list = [0.04, 0.12, 0.20];

data = [0.0029879 0.0067581 0.019039; 0.021031 0.035091 0.069893; 0.052391 0.076206 0.12903];
result = zeros(3);

for i = 1:3
    p_bar = p_bar_list(i);
    for j = 1:3
        rho = rho_list(j);
        
        fun = @(z) (normcdf((norminv(p_bar) + sqrt(rho)*z) / sqrt(1-rho)))^2 * normpdf(z);
        integral_value = integral(fun, -inf, inf, 'ArrayValued', true);
        result(i,j) = (integral_value - p_bar^2) / (p_bar - p_bar^2);
    end
end

result

%% 4.5
clear
clc
p_bar_list = [0.04, 0.12, 0.20];
rhox_matrix = [0.03614, 0.1343, 0.4541; 0.06279, 0.1960, 0.5255; 0.07745, 0.2263, 0.5565];
obligors = 1000;
loan_loss = 0.6;
m = 1000;
a = zeros(3);
b = zeros(3);
rho_list = [0.15, 0.40, 0.80; 0.15, 0.40, 0.80; 0.15, 0.40, 0.80];
alpha_list = [0.95, 0.99, 0.999];

for i = 1:3
    p_bar = p_bar_list(i);
    for j = 1:3
        rhox = rhox_matrix(i,j);
        
        a(i,j) = p_bar*((1-rhox) / rhox);
        b(i,j) = (1 - p_bar) * ((1-rhox) / rhox);
        
        
    end
end

% Merton model
for i = 1:3
    alpha = alpha_list(i);
    for j = 1:3
        p_bar = p_bar_list(j);
        for k = 1:3
            rho = rho_list(j,k);

            F_inv = (normcdf(((sqrt(rho)*norminv(alpha))+norminv(p_bar))/(sqrt(1-rho))));
            VaR = loan_loss*m*F_inv;

            inv = @(alpha) (normcdf(((sqrt(rho)*norminv(alpha))+norminv(p_bar))/(sqrt(1-rho))));

            ES = ((loan_loss*m)/(1-alpha))*integral(inv, alpha, 1);

            merton_result_table_var(i, j, k) = VaR/1000;
            merton_result_table_es(i, j, k) = ES/1000;


        end
    end
end
merton_result_table_var;
merton_result_table_es;

% Beta
beta_95_var = betainv(0.95, a, b) * 0.6;
beta_99_var = betainv(0.99, a, b) * 0.6;
beta_999_var = betainv(0.999, a, b) * 0.6;

beta_es = zeros(3,3,3);

for i = 1:3
    alpha = alpha_list(i);
    for j = 1:3
        for k = 1:3
            inv = @(alph) betainv(alph, a(j,k), b(j,k));
            beta_es(i,j,k) = (0.6 / (1-alpha)) * integral(inv, alpha, 1);
        end
    end
end
beta_es

%% Graph 4.5
X = (0:0.01:1);
for i = 1:3
    for j = 1:3
        plot(X, betapdf(X, a(i,j), b(i,j)), 'LineWidth', 1.8, 'DisplayName',['Curve #' num2str(i)] )
        hold on
    end
end

labels = {'(1,1)', '(1,2)', '(1,3)', '(2,1)', '(2,2)', '(2,3)', '(3,1)', '(3,2)', '(3,3)'};

legend(labels,'Location','northeast', 'FontSize', 12)
hold off

%% 4.6
clc
clear
a = [0.5, 1, 2, 5, 30];
b = [4.5, 9, 18, 45, 270];
m = 1000;
loss = 0.6;

result_table_var = loss * betainv(0.99, a, b)

for i = 1:5 
    inv = @(alpha) betainv(alpha, a(i), b(i));
    result_table_es(i) = (0.6 / 0.01) * integral(inv, 0.99, 1);
end

result_table_es

for i = 1:5
    p_bar(i) = a(i) / (a(i) + b(i));
    rho(i) = 1 / (a(i) + b(i) + 1);
end

p_bar
rho

%% GRAPH 4.6
x = 0:0.001:1;
labels = {'\lambda = 0.5', '\lambda = 1', '\lambda = 2', '\lambda = 5', '\lambda = 30'};
plot(betapdf(x, 0.5, 4.5), 'LineWidth', 2);
hold on
plot(betapdf(x, 1, 9), 'LineWidth', 2);
hold on
plot(betapdf(x, 2, 18), 'LineWidth', 2);
hold on
plot(betapdf(x, 5, 45), 'LineWidth', 2);
hold on
plot(betapdf(x, 30, 270), 'LineWidth', 2);
xlim([0, 400])
legend(labels,'Location','northeast', 'FontSize', 12)

title('Distributions for different values of \lambda', 'FontSize', 12)

hold off
%%
function [s,b] = bs(v,d,t,terminal,sigma,r)

    d1 = (log(v/d) + (r+(0.5*(sigma^2)))*(terminal - t))/(sigma * sqrt(terminal - t));
    
    d2 = d1 - (sigma * sqrt(terminal - t));
    
    s = v * normcdf(d1) - d*exp(-r*(terminal - t))*normcdf(d2);
    
    b = v - s;
end

function s = spread(v, d, t, terminal, sigma, r)
    [a, b] = bs(v, d, t, terminal, sigma, r);
    
    yield = (1/(terminal - t)) * log(d/b);
    
    s = yield - r;
end

function pd = defprob(v, d, mu, sigma, t)
    numerator = log(d/v) - (mu - (0.5*(sigma^2)))*t;
    denominator = sigma*sqrt(t);
    pd = normcdf(numerator/denominator);
end

% Sum all discounted and risk-adjusted coupons and then add discounted and
% risk-adjusted principal repayment.
function dcb = DefaultableCouponBond(lambdas, period_t, coupon_times, coupons, r)
    % ADD 0 TO START OF PERIOD_T SO THAT INDEXING WORKS WITHOUT CRASH
    lambdas = [0, lambdas];
    period_t = [0, period_t];
    
    % ADD PRINCIPAL TO LAST COUPON
    coupons(length(coupons)) = coupons(length(coupons)) + 1;
    
    % GO THROUGH ALL COUPONS TO ADJUST VALUATION FOR INTEREST AND RISK
    for i = 1: length(coupons)
        
        % FIND LOWER TIME BOUND FOR INTENSITY INTERVAL
        index_floor = min(find(period_t >= coupon_times(i)) - 1);
        
        % CALCULATE SURVIVAL PROBABILITY START AT 2 BECAUSE ADDED 0 ABOVE
        exp_sum = 0;
        for j = 2:index_floor
            exp_sum = exp_sum + lambdas(j) * (period_t(j) - period_t(j-1));
        end
        exp_sum = - exp_sum - (coupon_times(i) - period_t(index_floor)) * lambdas(index_floor + 1); 
        coupons(i) = coupons(i) * exp(-r * coupon_times(i)) * (1-(1-exp(exp_sum)));
        
    end
    dcb = sum(coupons);
end
    
function v = Vasicek(coupons, coupon_times, mu, vol, r, alpha, principal, lambda)
    % ADD PRINCIPAL TO LAST COUPON
    coupons(length(coupons)) = coupons(length(coupons)) + principal;
    
    % GO THROUGH ALL COUPONS TO ADJUST VALUATION FOR INTEREST AND RISK
    for i = 1: length(coupons)
        
        B = (1 - exp(-alpha*(coupon_times(i)))) / alpha;
    
        A = ((B - (coupon_times(i))) * (mu - (vol^2 / (2*(alpha^2))))) - (((vol^2)/(4*alpha))*B^2);
        
        % p = exp((B * ((mu/alpha)-(0.5*(vol/alpha)^2)-lambda) - (coupon_times(i)*((mu/alpha)-0.5*(vol/alpha)^2)) - (vol^2/(4*(alpha^3)))*(1-exp(-alpha*coupon_times(i)))^2))
        % exp(A - B*lambda)
        coupons(i) = coupons(i) * exp(-r * coupon_times(i)) * exp(A - B*lambda);
        % coupons(i) = coupons(i) * exp(-r * coupon_times(i)) * p;
        
    end
    v = sum(coupons);
end

