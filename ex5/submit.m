function submit()
  addpath('../lib','../lib/jsonlab');

  conf.assignmentKey = '-wEfetVmQgG3j-mtasztYg';
  conf.itemName = 'Regularized Linear Regression and Bias/Variance';
  conf.partArrays = { ...
    { ...
      'a6bvf', ...
      { 'linearRegCostFunction.m' }, ...
      'Regularized Linear Regression Cost Function', ...
    }, ...
    { ...
      'x4FhA', ...
      { 'linearRegCostFunction.m' }, ...
      'Regularized Linear Regression Gradient', ...
    }, ...
    { ...
      'n3zWY', ...
      { 'learningCurve.m' }, ...
      'Learning Curve', ...
    }, ...
    { ...
      'lLaa4', ...
      { 'polyFeatures.m' }, ...
      'Polynomial Feature Mapping', ...
    }, ...
    { ...
      'gyJbG', ...
      { 'validationCurve.m' }, ...
      'Validation Curve', ...
    }, ...
  };
  conf.output = @output;

  submitWithConfiguration(conf);
%  rmpath('../lib/jsonlab', '../lib'); 
end

function out = output(partId, auxstring)
  % Random Test Cases
  X = [ones(10,1) sin(1:1.5:15)' cos(1:1.5:15)'];
  y = sin(1:3:30)';
  Xval = [ones(10,1) sin(0:1.5:14)' cos(0:1.5:14)'];
  yval = sin(1:10)';
  if partId == 'a6bvf'
    [J] = linearRegCostFunction(X, y, [0.1 0.2 0.3]', 0.5);
    out = sprintf('%0.5f ', J);
  elseif partId == 'x4FhA'
    [J, grad] = linearRegCostFunction(X, y, [0.1 0.2 0.3]', 0.5);
    out = sprintf('%0.5f ', grad);
  elseif partId == 'n3zWY'
    [error_train, error_val] = ...
        learningCurve(X, y, Xval, yval, 1);
    out = sprintf('%0.5f ', [error_train(:); error_val(:)]);
  elseif partId == 'lLaa4'
    [X_poly] = polyFeatures(X(2,:)', 8);
    out = sprintf('%0.5f ', X_poly);
  elseif partId == 'gyJbG'
    [lambda_vec, error_train, error_val] = ...
        validationCurve(X, y, Xval, yval);
    out = sprintf('%0.5f ', ...
        [lambda_vec(:); error_train(:); error_val(:)]);
  end 
end
