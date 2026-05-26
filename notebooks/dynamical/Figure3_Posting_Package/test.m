% test_setup.m
% A quick script to verify MATLAB execution and variable tracking in VS Code

% 1. Generate some sample arrays
time_steps = linspace(0, 2*pi, 100);
signal = sin(time_steps);

% 2. Perform a basic calculation
signal_mean = mean(signal);

% 3. Create a simple plot
figure;
plot(time_steps, signal, 'LineWidth', 2);
title('VS Code MATLAB Integration Test');
xlabel('Time');
ylabel('Amplitude');

% 4. Print output to the terminal
disp(['Test complete! The mean of the signal is: ', num2str(signal_mean)]);
disp('Check your debugger variables panel for time_steps, signal, and signal_mean.');