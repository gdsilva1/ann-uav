function [eta,m,l,lambda,Ixx,Iyy,Izz,g,M,kf,km,Omegar,Cdj,Aj]=parametros

linear_vel=[0,0,0]';                                                % [x_dot,y_dot,z_dot];
ang_vel=[0,0,0]';                                                   % [phi_dot,theta_dot,psi_dot];
linear_pos=[0,0,0]';                                                % [x,y,z];
ang_pos=[0,0,0.5]';                                                 % [phi,theta,psi];      

eta=[linear_vel;ang_vel;linear_pos;ang_pos];                        % initial condition

m = 2.2;                                                            % mass of the quadcopter;
l = 0.1725;                                                         % distance between the motors and center of mass;
lambda = 1;                                                         % relation bewteen force and torque;
Ixx = 0.0167;                                                       % moment of inertia in x direction;
Iyy = 0.0167;                                                       % moment of inertia in y direction;
Izz = 0.0231;                                                       % moment of inertia in z direction;
g = 9.81;                                                           % acceleration due to gravity;
kf=3.13e-5;
km=7.50e-7;
Omegar=0;
Cdj=0.5;    
Aj=0.202;

M = zeros(6,6);

M(1,1) = m;
M(2,2) = m;
M(3,3) = m;
M(4,4) = Ixx;
M(5,5) = Iyy;
M(6,6) = Izz;

end
