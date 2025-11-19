function evaluate(x, al::AL)
   c = al.C.c(x)
   return al.f(x) - dot(c, al.lambda) + (0.5*al.mu) * sum(abs2, c)
end

function gradient!(out, x, al::AL)
   # Compute the constraint and its jacobian
   c = al.C.cJc!(x, al.Dc)
   
   # Evaluate gradient of f
   al.df(out, x)
   
   # Add penalty and Lagrange multiplier terms
   for i = 1:length(c)
       for j = 1:length(out)
           out[j] = out[j] - al.lambda[i] * al.Dc[i, j] + (1.0*al.mu) * c[i] * al.Dc[i, j]
       end
   end
   return out
end

function eval_and_grad!(out, x, al::AL)
   c = al.C.cJc!(x, al.Dc)
   f = al.fdf(out, x)
   for i = 1:length(c)
       for j = 1:length(out)
           out[j] = out[j] - al.lambda[i] * al.Dc[i, j] + (1.0*al.mu) * c[i] * al.Dc[i, j]
       end
   end
   return f - dot(c, al.lambda) + (0.5*al.mu) * sum(abs2, c)
end

gradient(x, al::AL) = gradient!(zeros(length(x)), x, al)

