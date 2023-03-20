from .base import GPTBase

def get_model(args):
    """ Return the right model """
    if args.model == 'base':
        model = GPTBase(args)
        if args.use_pretrained != 'none':
            raise NotImplementedError(f"Loading of pretrained models not yet implemented for model '{args.model}'.") 
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")