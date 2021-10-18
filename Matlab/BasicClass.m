classdef BasicClass
   properties(Constant)
      Value=2;
   end
   methods(Static=true)
      function r = roundOff()
         r = round([BasicClass.Value],2);
      end
      function r = multiplyBy(n)
         r = [BasicClass.Value] * n;
      end
   end
end