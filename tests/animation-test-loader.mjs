// Use the deployed browser dependency in Node without changing production imports.
// node --import ./tests/animation-test-loader.mjs --test tests/*.test.mjs
import {registerHooks} from 'node:module';
import {pathToFileURL} from 'node:url';

registerHooks({
  resolve(specifier,context,nextResolve){
    if(specifier==='/animation/vendor/three.module.js'&&process.env.ANIMATION_THREE_MODULE){
      return {url:pathToFileURL(process.env.ANIMATION_THREE_MODULE).href,shortCircuit:true};
    }
    return nextResolve(specifier,context);
  }
});
