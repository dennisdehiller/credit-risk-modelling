%% 1.1
clc
[s,b] = bs(135, 95, 0, 2, 0.2, 0.02);

%% 1.2
clc
list95 = (0.05:0.01:0.3);
list70 = (0.05:0.01:0.3);
for i = 1:length(list95)
    list95(i) = spread(110, 95, 0, 2, list95(i), 0.03);
    list70(i) = spread(110, 70, 0, 2, list70(i), 0.03);
end

vol = (0.05:0.01:0.3);

plot(vol, list95);
hold on
ylim([0, 0.07])
title('Yield spread as a function of volatility: V=110, T=2, r=0.02', 'FontSize', 16)
xlabel('\sigma', 'FontSize', 15) 
ylabel('Yield Spread', 'FontSize', 15) 
plot(vol, list70);
legend({'D = 95','D = 70'},'Location','northeast', 'FontSize', 12)
%print -deps 1.2.eps

%% 1.3
clc
senior = 150 - bs(150, 75, 0, 3, 0.35, 0.02);
equity = bs(150, 135, 0, 3, 0.35, 0.02);
junior = bs(150, 75, 0, 3, 0.35, 0.02) - equity;
total = senior + equity + junior;

%% 1.4
clc
mu_1yr = (0:0.02:0.5);
for i = 1:length(mu_1yr)
    mu_1yr(i) = defprob(200, 150, mu_1yr(i), 0.3, 5);
end

xval = (0:0.02:0.5);
plot(xval, mu_1yr);
title('The default probability in the Merton model: D=150, V=200, \sigma=0.3', 'FontSize', 16)
xlabel('\mu', 'FontSize', 15) 
ylabel('Default probability', 'FontSize', 15) 
print -deps 1.4.eps

mertonmodel(200, 0.3, 150, 0, 'Drift',0.3, 'Maturity',5)
pd = defprob(200, 150, 0.3, 0.3, 5);

%% 1.5
clc

df_prob_1yr = xlsread("blm.xlsx", 'bloomberg', 'A2');
df_prob_2yr = xlsread("blm.xlsx", 'bloomberg', 'B2');
df_prob_3yr = xlsread("blm.xlsx", 'bloomberg', 'C2');
df_prob_4yr = xlsread("blm.xlsx", 'bloomberg', 'D2');
df_prob_5yr = xlsread("blm.xlsx", 'bloomberg', 'E2');
totalt_debt = xlsread("blm.xlsx", 'bloomberg', 'F2');
vol_1yr = xlsread("blm.xlsx", 'bloomberg', 'G2');
mkt_cap = xlsread("blm.xlsx", 'bloomberg', 'H2');
debt_equity_ratio = xlsread("blm.xlsx", 'bloomberg', 'I2');


T_1 = 1;
T_3 = 3;
vol_3yr = vol_1yr*sqrt(T_3);

mu_1yr = log(debt_equity_ratio)/T_1+0.5*vol_1yr*(vol_1yr-(2*norminv(df_prob_1yr))/sqrt(T_1));
DD_1yr = (log(debt_equity_ratio) - (mu_1yr-0.5*(vol_1yr^2))*T_1) / vol_1yr*sqrt(T_1);
drsk_1yr = normcdf(DD_1yr);

mu_3yr = log(debt_equity_ratio)/T_3+0.5*vol_3yr*(vol_3yr-(2*norminv(df_prob_3yr))/sqrt(T_3));
vol_3yr;
mu_3yr;
DD_3yr = (log(debt_equity_ratio) - (mu_3yr-0.5*(vol_3yr^2))*T_3) / vol_3yr*sqrt(T_3);
drsk_3yr = normcdf(DD_3yr);

drsk_3yr;
df_prob_3yr;


%% 1.6
clc




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
table;
phi5 = table(1,:);
phi40 = table(3,:);

for i = 1:(length(phi5) - 1)
    phi5(i) = table(1, (i)) - table(1, i+1);
    phi40(i) = table(3, (i)) - table(3, i+1);
end

phi5(1:9);
phi40(1:9);

%% 3.1
clc
p = [0.5 0.35 0.10 0.05; 0.35 0.3 0.25 0.10; 0.35 0.25 0.25 0.15; 0 0 0 1];
a = [0.4 0.5 0.1 0];

% a
p2 = a * (p^2);
p3a = a * (p^3);
p7 = a * (p^7);

% b
invp1 = 1-(a * p);
invp4 = 1-(a * p^4);
invp6 = 1-(a * p^6);

% c
p3 = a * (p^3);
p3c = p3(3) + p3(4);

% d
a = [0 1 0 0];

p3d = a * (p^3);

%% 3.2
clc
Q = [-0.1055 0.0704 0.0351; 0.242 -0.329 0.087; 0 0 0];
a = [0.5 0.5 0];

% a
p2a = a * (expm(Q*2));
p3a = a * (expm(Q*3));
p7a = a * (expm(Q*7));

% b
p1inv = 1 - (a * (expm(Q)));
p4inv = 1 - (a * (expm(Q * 4)));
p6inv = 1 - (a * (expm(Q * 6)));

% c
p35 = a * (expm(Q*3.5));
p35 = p35(2) + p35(3);

% d
a = [0 1 0];
p3d = a * (expm(Q*3));

%% 3.3
clc
clear
Q = [-0.1055, 0.0704, 0.0351; 0.242, -0.329, 0.087; 0, 0, 0];
alpha_aaa = [1, 0, 0];
alpha_aa = [0, 1, 0];

T_1 = [1, 3, 5, 7, 10];
rf = 0.03;

prob_aaa = zeros(5, 3);
prob_aa = zeros(5, 3);

for i = 1:5
    prob_aaa(i,:) = alpha_aaa * (expm(T_1(i) * Q));
end

for i = 1:5
    prob_aa(i,:) = alpha_aa * (expm(T_1(i) * Q));
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

price_aaa_10;
price_aaa_50;
price_aa_10;
price_aa_50;

%% 4.1
clc
clear

%using eq 5.9.25 (s109)-      fråga vad skillnaden på den och 5.9.1.5 är.
%Är den mer precis?

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

    

