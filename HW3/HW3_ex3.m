clear all
close all

%% Pricing the Merton Jump-Diffusion option for different strikes and maturities.
N=10;
lambda=1;
gamma=-.1;
sigma=.2;
q=0;
r=.04;
S_0=100;
T=[.02 .08];
dK=.01;
strikes=60-dK:dK:150+dK;
length(strikes)
option_prices=zeros(length(strikes),length(T));
for j=1:length(strikes)
    for i=1:length(T)
        option_prices(j,i)=JMoptionprice(S_0,gamma,strikes(j),sigma,r,T(i),lambda,N);
    end
end
option_prices;

%% Numerical approximation of second derivative, which gives us the implied PDF of the stock
implied_pdf=zeros(length(strikes)-1,length(T));
for i=2:length(strikes)-1
    implied_pdf(i,:)=(option_prices(i+1,:)-2*option_prices(i,:)+option_prices(i-1,:))/dK^2;
end

%% Plotting
plot(strikes(2:end),implied_pdf(:,1))
hold on
plot(strikes(2:end),implied_pdf(:,2))
title('Implied probability distribution of $S$',Interpreter='latex')
xlabel('$S$',Interpreter='latex')
ylabel('Density')
legend('T=0.02','T=0.08')


%% Pricing functions
function jump_price=JMoptionprice(S,gamma,K,sigma,r,T,lambda,N)
price=0;
for j=0:N
    P=exp(-lambda*T)*(lambda*T)^j/factorial(j);
    price=price+P*BSoptionprice(S*(1+gamma)^j,K,sigma,r,T,lambda*gamma);
end
jump_price=price;
end

function price=BSoptionprice(S,K,sigma,r,T,q)
d_1=(log(S/K)+(r-q+0.5*sigma^2)*T)/(sqrt(T)*sigma);
d_2=d_1-sqrt(T)*sigma;
price=exp(-q*T)*S*normcdf(d_1)-K*exp(-r*T)*normcdf(d_2);
end
