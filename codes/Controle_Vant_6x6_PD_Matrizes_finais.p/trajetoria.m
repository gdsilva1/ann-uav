function [x_des,y_des,z_des,x_dot_des,y_dot_des,z_dot_des]=trajetoria(t,option)

if option == 1
    
velo=0.4;

if t<8
    x_des=velo*t;
    y_des=0;
    z_des=2;
    x_dot_des=0;
    y_dot_des=0;
    z_dot_des=0;
elseif t<16
    x_des=velo*8;
    y_des=velo*(t-8);
    z_des=2;
    x_dot_des=0;
    y_dot_des=0;
    z_dot_des=0;
elseif t<24
    x_des=velo*8-velo*(t-16);
    y_des=velo*8;
    z_des=2;
    x_dot_des=0;
    y_dot_des=0;
    z_dot_des=0;
elseif t<32
    x_des=0;
    y_des=velo*8-velo*(t-24);
    z_des=2;
    x_dot_des=0;
    y_dot_des=0;
    z_dot_des=0;
else
    x_des=0;
    y_des=0;
    z_des=2;
    x_dot_des=0;
    y_dot_des=0;
    z_dot_des=0;
end

elseif option ==2
    
    x_des=0.5*cos(t/2);
    y_des=0.5*sin(t/2);
    z_des=3*ones(1,length(t));
%     z_des=1+t/10;
    x_dot_des=-0.25*sin(t/2);
    y_dot_des=0.25*cos(t/2);
    z_dot_des=0*ones(1,length(t));
    
else
    
    % Sobe/cruzeiro/desce(ou não)
    x_des=[zeros(1,round(length(t)/5)),2*ones(1,length(t)-round(2*length(t)/5)),zeros(1,round(length(t)/5))];
    y_des=[zeros(1,round(length(t)/5)),2*ones(1,length(t)-round(2*length(t)/5)),zeros(1,round(length(t)/5))];
    z_des=[zeros(1,round(length(t)/5)),2*ones(1,length(t)-round(2*length(t)/5)),zeros(1,round(length(t)/5))];
    x_dot_des=zeros(1,length(t));
    y_dot_des=zeros(1,length(t));
    z_dot_des=zeros(1,length(t));

end

end