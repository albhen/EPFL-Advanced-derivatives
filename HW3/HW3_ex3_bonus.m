clear all
close all

%% Generating stock prices at T to obtain an empirical PDF.
lambda=1;
gamma=-.1;
sigma=.2;
q=0;
r=.04;
S_0=100;
T=[.02 .08];
N=10^7;
S_T1=zeros(N,1);
S_T2=zeros(N,1);

% Standard BM at maturity.
W_T1=normrnd(0,1,[N,1])*sqrt(T(1));
W_T2=normrnd(0,1,[N,1])*sqrt(T(2));

% Poisson distributed number of jumps.
N_T1=poissrnd(lambda*T(1),[N,1]);
N_T2=poissrnd(lambda*T(2),[N,1]);

S_T1=S_0*exp((r-q-lambda*gamma-0.5*sigma^2)*T(1)+sigma*W_T1).*(1+gamma).^N_T1;
S_T2=S_0*exp((r-q-lambda*gamma-0.5*sigma^2)*T(2)+sigma*W_T2).*(1+gamma).^N_T2;

% Empirical density.
[density1,x]=ksdensity(S_T1);
[density2,y]=ksdensity(S_T2);

%% Plotting
plot(x,density1)
hold on 
plot(y,density2)
title('Empirical distribution of $S$',Interpreter='latex')
xlabel('$S$',Interpreter='latex')
ylabel('Density')
legend('T=0.02','T=0.08')
