import { Link, useLocation } from 'react-router';
import { APP_PRIMARY_NAV } from '../../lib/app-nav';

export function Navigation() {
  const location = useLocation();

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  return (
    <header className="fixed top-0 left-0 w-full z-50 px-[48px] py-4">
      <nav className="bg-[rgba(13,21,21,0.65)] backdrop-blur-[20px] border border-white/10 rounded-lg flex justify-between items-center px-6 py-3 shadow-[0_4px_30px_rgba(0,0,0,0.1)]">
        <div className="flex items-center gap-4">
          <Link to="/" className="font-bold text-[32px] leading-[40px] tracking-tight text-[#00dbe9]">
            QuLab Infinite
          </Link>
          <div className="h-4 w-[1px] bg-[#3b494b]/30 hidden lg:block" />
          <div className="hidden lg:flex gap-4 flex-wrap">
            {APP_PRIMARY_NAV.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`text-[11px] leading-[16px] tracking-[0.12em] font-bold uppercase whitespace-nowrap ${
                  isActive(item.path)
                    ? 'text-[#00dbe9] border-b-2 border-[#00dbe9] pb-1'
                    : 'text-[#b9cacb] hover:text-[#dbfcff]'
                } transition-all duration-200`}
              >
                {item.label}
              </Link>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-4">
          <Link
            to="/labs"
            className="bg-[#00f0ff] text-[#00363a] text-[12px] leading-[16px] tracking-[0.15em] font-bold px-6 py-2 rounded-sm hover:opacity-90 active:scale-95 transition-all uppercase"
          >
            INITIALIZE SYSTEM
          </Link>
        </div>
      </nav>
    </header>
  );
}
