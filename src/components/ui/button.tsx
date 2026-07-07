import * as React from "react"
import { cn } from "./card"

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'ghost' | 'destructive';
  size?: 'default' | 'sm' | 'lg' | 'icon';
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={cn(
          "inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-500 disabled:pointer-events-none disabled:opacity-50",
          {
            'bg-blue-600 text-white hover:bg-blue-500 shadow-sm': variant === 'default',
            'border border-white/10 bg-transparent hover:bg-white/5 text-zinc-300': variant === 'outline',
            'hover:bg-white/5 hover:text-zinc-100 text-zinc-400': variant === 'ghost',
            'bg-red-900/50 text-red-200 hover:bg-red-900 border border-red-800': variant === 'destructive',
            'h-9 px-4 py-2': size === 'default',
            'h-8 rounded-md px-3 text-xs': size === 'sm',
            'h-10 rounded-md px-8': size === 'lg',
            'h-9 w-9': size === 'icon',
          },
          className
        )}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button }
