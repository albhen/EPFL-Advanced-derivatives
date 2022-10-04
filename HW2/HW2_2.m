clear all
close all

%% Implied volatility of the Jump-diffusion model (constant jumps)
S=100;
gamma=-.07;
lambdaQ=.25;
sigma=.2;
r=.05;
T=[.02 .08 .25 .5];
K=S*[.8 .9 1 1.1];
N=10;
price=zeros(length(T),length(K));
for i=1:length(T)
    for k=1:length(K)
        for j=0:N
            P=exp(-lambdaQ*T(i))*(lambdaQ*T(i))^j/factorial(j);
            price(i,k)=price(i,k)+P*BSoptionprice(S*(1+gamma)^j,K(k),sigma,r,T(i),lambdaQ*gamma);
        end
    end
end

implied_volatility=zeros(length(T),length(K));
startguess_V=sigma;
for i=1:length(T)
    for j=1:length(K)
        implied_volatility(i,j)=newton_rhapson_vol(S,K(j),price(i,j),T(i),r,startguess_V);
    end
end

%% Plot  IV vs strike for each maturity 

for i=1:length(T)
    subplot(2,2,i)
    plot(K,implied_volatility(i,:))
    xlabel('Strike')
    ylabel('Implied volatility')
    title('$T$='+string(T(i)),'Interpreter','latex')
    grid on
end


%% BS option pricing formula

function price=BSoptionprice(S,K,sigma,r,T,q)
d_1=(log(S/K)+(r-q+0.5*sigma^2)*T)/(sqrt(T)*sigma);
d_2=d_1-sqrt(T)*sigma;
price=exp(-q*T)*S*normcdf(d_1)-K*exp(-r*T)*normcdf(d_2);
end

%% Newton-Rhapson for implied volatility

function solution=newton_rhapson_vol(S,K_1,K_2,tau,r,startguess)
x = startguess;
x_old=100;
iter = 0;
while abs(x_old-x) > 1e-8  
    x_old = x;  
    x = x - (S*normcdf((log(S/K_1)+(r+x^2/2)*(tau))/(sqrt(tau)*x))-K_1*exp(-r*(tau))*normcdf((log(S/K_1)+ ...
        (r-x^2/2)*(tau))/(sqrt(tau)*x))-K_2)/(sqrt(tau)*S*normpdf((log(S/K_1)+(r+x^2/2)*(tau))/(sqrt(tau)*x)));  
end 
solution=x;
end