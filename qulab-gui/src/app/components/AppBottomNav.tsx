import { useNavigate, useLocation } from 'react-router';
import { APP_BOTTOM_NAV_OS } from '../../lib/app-nav';

/** Fixed bottom OS nav — matches Figma Make medical/dashboard chrome. */
export function AppBottomNav({ className = '' }: { className?: string }) {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <nav
      className={`fixed bottom-0 left-0 w-full z-50 flex justify-around items-stretch h-16 bg-surface-container-lowest/90 backdrop-blur-md border-t border-outline-variant/50 ${className}`}
    >
      {APP_BOTTOM_NAV_OS.map((item) => {
        const active = location.pathname === item.path || location.pathname.startsWith(item.path + '/');
        return (
          <button
            key={item.path}
            type="button"
            onClick={() => navigate(item.path)}
            className={`flex flex-col items-center justify-center py-unit px-4 transition-all duration-150 ${
              active
                ? 'text-primary-fixed-dim bg-primary-container/20 border-t-2 border-primary-fixed-dim'
                : 'text-on-surface-variant/60 hover:text-primary-fixed-dim hover:bg-surface-variant/30'
            }`}
          >
            <span
              className="material-symbols-outlined"
              style={active ? { fontVariationSettings: '"FILL" 1' } : undefined}
            >
              {item.icon}
            </span>
            <span className="font-label-caps text-label-caps">{item.label.toUpperCase()}</span>
          </button>
        );
      })}
    </nav>
  );
}
